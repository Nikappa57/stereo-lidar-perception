"""Preprocessing functions for stereo-lidar perception.


Four representations are provided:

* **BEV**: Bird's-Eye-View density / height map projected from the LiDAR
  point cloud onto a regular 2-D grid.
* **Frustum**: Camera-frustum point cloud: LiDAR returns that fall inside a
  2-D detection box are lifted into a local frustum frame, optionally with
  RGB colour from the left image appended.
* **Voxel**: Volumetric occupancy / feature grid built from the ego-frame
  point cloud with configurable resolution and feature channels.
* **Clustering**: Euclidean-distance DBSCAN clustering of the point cloud
    returning per-point labels and per-cluster stats.

Quick start::

    from data import Py123dDataset
    from preprocessing import bev_map, frustum_points, voxel_grid, cluster_points

    dataset = Py123dDataset(split_names=["av2-sensor_val"])
    sample  = dataset[0].to_stereo_sample()

    bev  = bev_map(sample)               # (C, H_bev, W_bev)
    fpts = frustum_points(sample, box_index=0)   # (P, 6)  XYZ+RGB
    vox  = voxel_grid(sample)            # (C, D, H, W)
    lbls, stats = cluster_points(sample) # (P,) int, list[ClusterStats]
"""


from __future__ import annotations

import sysconfig

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.nn as nn
from matplotlib import streamplot


from data import Calibration, StereoSample
from monobev import MonoBEVConfig, _EfficientNetBackbone, build_frustum_points, splat
from pointpillars import PillarConfig, pillarize, PillarFeatureNet, PointPillarsScatter, BEVBackbone2D
from stereo import StereoSGBMConfig
from stereobev import StereoBEVConfig, StereoBEV

# Helpers for preprocessing

def _ego_to_cam(
    xyz_ego: np.ndarray,
    cam_to_ego: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform ego-frame points into camera coordinates.

    :param xyz_ego:    ``(P, 3)`` points in the **ego** frame.
    :param cam_to_ego: ``(4, 4)`` camera → ego transform (from Calibration).
    :returns: ``(xyz_cam, ego_to_cam_4x4)`` — camera-frame points ``(P, 3)``
              and the 4×4 inverse transform used internally.
    """
    ego_to_cam = np.linalg.inv(cam_to_ego)
    pts_h = np.hstack([xyz_ego, np.ones((len(xyz_ego), 1), dtype=xyz_ego.dtype)])
    xyz_cam = (ego_to_cam @ pts_h.T).T[:, :3]
    return xyz_cam, ego_to_cam

def _cam_to_ego(
    xyz_cam: np.ndarray,
    cam_to_ego: np.ndarray,
) -> np.ndarray:
    """Transform camera-frame points into ego coordinates.

    :param xyz_cam: ``(P, 3)`` points in the **camera** frame.

    :param cam_to_ego: ``(4, 4)`` camera → ego transform (from Calibration).
    :returns: ``(xyz_ego, ego_to_cam_4x4)`` — ego-frame points ``(P, 3)``
              and the 4×4 inverse transform used internally.
    """
    pts_h = np.hstack([xyz_cam, np.ones((len(xyz_cam), 1), dtype=xyz_cam.dtype)])
    xyz_ego = (cam_to_ego @ pts_h.T).T[:, :3]
    return xyz_ego


def _project_to_image(
    xyz_cam: np.ndarray,
    K: np.ndarray,
    img_w: int,
    img_h: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pinhole projection + depth & image-boundary filter.

    :param xyz_cam: ``(P, 3)`` points in camera frame.
    :param K:       ``(3, 3)`` intrinsics matrix.
    :returns: ``(uv, mask)`` — pixel coords ``(M, 2)`` (float) and boolean
              mask of length P selecting the M valid points.
    """
    in_front = xyz_cam[:, 2] > 0.0
    depth    = np.where(in_front, xyz_cam[:, 2], 1.0)  # avoid div-by-zero
    uv       = (K @ xyz_cam.T).T
    uv       = uv[:, :2] / depth[:, None]
    valid    = (
        in_front
        & (uv[:, 0] >= 0) & (uv[:, 0] < img_w)
        & (uv[:, 1] >= 0) & (uv[:, 1] < img_h)
    )
    return uv, valid

# 1.  BEV  

@dataclass
class PointPillarsBranch(nn.Module):
    """End-to-end module: raw point cloud -> BEV LiDAR feature map."""

    def __init__(self, cfg: PillarConfig, pillar_feat_channels: int = 64, use_backbone: bool = True):
        super().__init__()
        self.cfg = cfg
        self.pfn = PillarFeatureNet(in_channels=9, out_channels=pillar_feat_channels)
        self.scatter = PointPillarsScatter(cfg.grid_size, channels=pillar_feat_channels)
        self.use_backbone = use_backbone
        if use_backbone:
            self.backbone = BEVBackbone2D(in_channels=pillar_feat_channels, out_channels=128)

    def forward(self, points: np.ndarray, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        pillar_points, pillar_coords, npoints = pillarize(points, self.cfg)
        nx, ny = self.cfg.grid_size

        if pillar_points.shape[0] == 0:
            c = 128 if self.use_backbone else self.pfn.linear.out_features
            return torch.zeros(c, nx, ny, device=device)

        pillar_points_t = torch.from_numpy(pillar_points).to(device)
        pillar_coords_t = torch.from_numpy(pillar_coords).to(device)
        npoints_t = torch.from_numpy(npoints).to(device)

        pillar_feats = self.pfn(pillar_points_t, npoints_t)
        bev = self.scatter(pillar_feats, pillar_coords_t)

        if self.use_backbone:
            bev = self.backbone(bev.unsqueeze(0)).squeeze(0)
        return bev  # (C, nx, ny) ready for alignment/fusion with the BEV camera


# MonoBEV: Lift-Splat-Shoot monocular BEV pipeline

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
       **sort + cumulative-sum** pooling — no Python loops, no
       ``scatter_add`` (see :meth:`_splat`).

    The output ``(C, nx, ny)`` tensor is pixel-aligned with the LiDAR BEV
    produced by :class:`PointPillarsBranch`.

    Args:
        cfg:  :class:`MonoBEVConfig` instance.
    """

    def __init__(self, cfg: Optional[MonoBEVConfig] = None):
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


# Stereo BEV (splat)

class StereoBEVBranch(nn.Module):
    """End-to-end module: stereo image pair → BEV camera feature map.

    Mirrors the structure of :class:`PointPillarsBranch` and :class:`MonoBEV`
    but uses a **grounded stereo-depth splat** instead of a predicted depth
    distribution:

    1. Compute SGBM stereo depth (rectified frame) from the image pair.
    2. Resize the rectified left image + depth to the backbone input resolution.
    3. Scale intrinsics (from the rectified ``P1``) to the new resolution.
    4. Build the rectified-left → ego extrinsic (``T_left2ego @ R1^T``).
    5. Run :class:`~stereobev.StereoBEV` — backbone → context head →
       grounded back-projection → BEV splat → BEV CNN refinement.

    The output ``(C, nx, ny)`` tensor is pixel-aligned with
    :class:`PointPillarsBranch` and :class:`MonoBEV` outputs.

    Args
    ----
    cfg        : :class:`~stereobev.StereoBEVConfig` (defaults when ``None``).
    sgbm_cfg   : :class:`~stereo.StereoSGBMConfig`  (defaults when ``None``).
    target_hw  : ``(H, W)`` to resize images to before the backbone.
    """

    def __init__(
        self,
        cfg:       Optional[StereoBEVConfig]  = None,
        sgbm_cfg:  Optional[StereoSGBMConfig] = None,
        target_hw: Tuple[int, int] = (192, 640),
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

        Parameters
        ----------
        sample : :class:`~data.StereoSample` with ``image_left``,
                 ``image_right``, and ``calibration``.
        device : Target torch device.

        Returns
        -------
        bev : (C, nx, ny) float32 tensor on *device*.
        """
        import torchvision.transforms.functional as TF
        import torch.nn.functional as _F
        from stereo import stereo_depth as _stereo_depth

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


# 2.  Frustum

@dataclass
class FrustumConfig:
    """Configuration for :func:`frustum_points`."""
    append_rgb:         bool  = True    # append left-image RGB to XYZ
    depth_normalise:    bool  = True    # normalise Z by frustum depth range
    min_depth_m:        float = 0.5     # discard closer returns
    max_depth_m:        float = 70.0    # discard farther returns
    frustum_frame:      bool  = True    # rotate so centroid is at +Z axis


def frustum_points(
    sample:     StereoSample,
    box_index:  int = 0,
    config:     Optional[FrustumConfig] = None,
) -> np.ndarray:
    """Extract a camera-frustum point cloud for one 2-D detection box.

    Selects LiDAR returns whose left-image projection falls inside
    ``sample.boxes_2d_left[box_index]`` (xyxy) and packages them into a
    local frustum frame.  RGB is optionally appended from the left image.

    :param sample:    A :class:`~data.StereoSample`.
    :param box_index: Which row of ``sample.boxes_2d_left`` to use.
    :param config:    :class:`FrustumConfig`; uses defaults when ``None``.
    :returns: Float32 array ``(P, D)`` where D is 3 (XYZ) or 6 (XYZ+RGB).
              Returns an empty ``(0, D)`` array when no points fall inside.
    :raises IndexError: if ``box_index`` is out of range.
    """
    cfg  = config or FrustumConfig()
    calib = sample.calibration
    D    = 6 if cfg.append_rgb else 3

    if sample.boxes_2d_left.shape[0] == 0 or box_index >= sample.boxes_2d_left.shape[0]:
        raise IndexError(
            f"box_index={box_index} but sample has {sample.boxes_2d_left.shape[0]} 2D boxes."
        )

    box_xyxy = sample.boxes_2d_left[box_index]  # (4,) x1,y1,x2,y2
    x1, y1, x2, y2 = box_xyxy
    img_h, img_w = sample.image_left.shape[:2]

    xyz_ego = sample.lidar_xyz.astype(np.float32)
    if xyz_ego.shape[0] == 0:
        return np.zeros((0, D), dtype=np.float32)

    # --- ego → camera ---
    xyz_cam, _ = _ego_to_cam(xyz_ego, calib.left_to_ego)

    # --- depth filter ---
    depth_mask = (xyz_cam[:, 2] >= cfg.min_depth_m) & (xyz_cam[:, 2] <= cfg.max_depth_m)
    xyz_cam = xyz_cam[depth_mask]
    xyz_ego = xyz_ego[depth_mask]

    # --- project & image-box filter ---
    K = calib.left_intrinsics.astype(np.float32)
    uv, valid = _project_to_image(xyz_cam, K, img_w, img_h)
    in_box = (
        valid
        & (uv[:, 0] >= x1) & (uv[:, 0] <= x2)
        & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)
    )
    if not in_box.any():
        return np.zeros((0, D), dtype=np.float32)

    pts_cam = xyz_cam[in_box]   # (M, 3) camera frame
    uv_sel  = uv[in_box]        # (M, 2) pixel coords

    # --- optional frustum-local rotation ---
    if cfg.frustum_frame:
        centroid_dir = pts_cam.mean(axis=0)
        centroid_dir /= (np.linalg.norm(centroid_dir) + 1e-9)
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        v = np.cross(centroid_dir, z_axis)
        s = np.linalg.norm(v)
        c = float(np.dot(centroid_dir, z_axis))
        if s > 1e-6:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float32)
            R  = np.eye(3, dtype=np.float32) + vx + vx @ vx * ((1 - c) / (s * s))
        else:
            R = np.eye(3, dtype=np.float32) if c > 0 else -np.eye(3, dtype=np.float32)
        pts_cam = (R @ pts_cam.T).T

    # --- depth normalisation ---
    if cfg.depth_normalise:
        z_min, z_max = pts_cam[:, 2].min(), pts_cam[:, 2].max()
        dz = z_max - z_min if z_max > z_min else 1.0
        pts_cam = pts_cam.copy()
        pts_cam[:, 2] = (pts_cam[:, 2] - z_min) / dz

    # --- append RGB ---
    if cfg.append_rgb:
        u_idx = np.clip(np.round(uv_sel[:, 0]).astype(np.int32), 0, img_w - 1)
        v_idx = np.clip(np.round(uv_sel[:, 1]).astype(np.int32), 0, img_h - 1)
        rgb   = sample.image_left[v_idx, u_idx].astype(np.float32) / 255.0
        return np.hstack([pts_cam, rgb])  # (M, 6)

    return pts_cam  # (M, 3)


# 3.  Voxels

@dataclass
class VoxelConfig:
    """Configuration for :func:`voxel_grid`."""
    x_range:    Tuple[float, float] = (-40.0, 40.0)
    y_range:    Tuple[float, float] = (-40.0, 40.0)
    z_range:    Tuple[float, float] = (-3.0,   1.0)
    voxel_size: float               = 0.2      # metres per voxel (isotropic)
    #: Channels per voxel:
    #:  0 – binary occupancy
    #:  1 – mean height (normalised)
    #:  2 – point count (log1p-normalised)
    #:  3 – mean intensity (if available)
    num_channels: int = 4
    max_points_per_voxel: int = 35   # cap for mean-feature computation


@dataclass
class VoxelGrid:
    """Output of :func:`voxel_grid`."""
    features:   np.ndarray          # (C, D, H, W) float32
    coords:     np.ndarray          # (N_occupied, 3) int32  (d, h, w) indices
    config:     VoxelConfig


def voxel_grid(
    sample: StereoSample,
    config: Optional[VoxelConfig] = None,
) -> VoxelGrid:
    """Build a volumetric feature grid from the ego-frame point cloud.

    :param sample: A :class:`~data.StereoSample`.
    :param config: :class:`VoxelConfig`; uses defaults when ``None``.
    :returns: :class:`VoxelGrid` with ``features`` shaped ``(C, D, H, W)``.
    """
    cfg = config or VoxelConfig()
    vs  = cfg.voxel_size

    D = int(round((cfg.z_range[1] - cfg.z_range[0]) / vs))
    H = int(round((cfg.y_range[1] - cfg.y_range[0]) / vs))
    W = int(round((cfg.x_range[1] - cfg.x_range[0]) / vs))

    empty_coords = np.zeros((0, 3), dtype=np.int32)
    empty_feat   = np.zeros((cfg.num_channels, D, H, W), dtype=np.float32)

    xyz = sample.lidar_xyz.astype(np.float32)
    if xyz.shape[0] == 0:
        return VoxelGrid(features=empty_feat, coords=empty_coords, config=cfg)

    # --- spatial filter ---
    mask = (
        (xyz[:, 0] >= cfg.x_range[0]) & (xyz[:, 0] < cfg.x_range[1])
        & (xyz[:, 1] >= cfg.y_range[0]) & (xyz[:, 1] < cfg.y_range[1])
        & (xyz[:, 2] >= cfg.z_range[0]) & (xyz[:, 2] < cfg.z_range[1])
    )
    xyz = xyz[mask]

    intensity: Optional[np.ndarray] = None
    if sample.lidar_features is not None and cfg.num_channels >= 4:
        for key in ("intensity", "reflectance"):
            if key in sample.lidar_features:
                intensity = sample.lidar_features[key].astype(np.float32)[mask]
                break

    # --- voxel indices ---
    ix = np.floor((xyz[:, 0] - cfg.x_range[0]) / vs).astype(np.int32).clip(0, W - 1)
    iy = np.floor((xyz[:, 1] - cfg.y_range[0]) / vs).astype(np.int32).clip(0, H - 1)
    iz = np.floor((xyz[:, 2] - cfg.z_range[0]) / vs).astype(np.int32).clip(0, D - 1)

    feat = np.zeros((cfg.num_channels, D, H, W), dtype=np.float32)

    # Channel 0: occupancy
    feat[0, iz, iy, ix] = 1.0

    # Channel 1: mean height (normalised)
    z_norm  = ((xyz[:, 2] - cfg.z_range[0]) / (cfg.z_range[1] - cfg.z_range[0])).clip(0.0, 1.0)
    count   = np.zeros((D, H, W), dtype=np.float32)
    np.add.at(count,   (iz, iy, ix), 1.0)
    np.add.at(feat[1], (iz, iy, ix), z_norm)
    occ_mask = count > 0
    feat[1, occ_mask] /= count[occ_mask]

    # Channel 2: log density
    max_c = count.max() if count.max() > 0 else 1.0
    feat[2] = np.log1p(count) / np.log1p(max_c)

    # Channel 3: mean intensity
    if intensity is not None and cfg.num_channels >= 4:
        np.add.at(feat[3], (iz, iy, ix), intensity)
        feat[3, occ_mask] /= count[occ_mask]
        peak = feat[3].max()
        if peak > 0:
            feat[3] /= peak

    occupied_dhw = np.stack(np.where(feat[0] > 0), axis=1).astype(np.int32)
    return VoxelGrid(features=feat, coords=occupied_dhw, config=cfg)


# 4.  Clustering (DBSCAN)

@dataclass
class ClusterConfig:
    """Configuration for :func:`cluster_points`."""
    eps_m:          float = 0.5     # DBSCAN neighbourhood radius (metres)
    min_samples:    int   = 5       # minimum cluster size
    use_bev:        bool  = True    # cluster in 2-D (XY) instead of 3-D XYZ
    #: If True, cluster only background returns (points_outside_boxes_xyz).
    background_only: bool = False
    z_range:        Tuple[float, float] = (-3.0, 1.0)   # height pre-filter


@dataclass
class ClusterStats:
    """Per-cluster statistics returned by :func:`cluster_points`."""
    label:      int             # cluster id (0-based; -1 = noise)
    num_points: int
    centroid:   np.ndarray      # (3,) ego-frame XYZ
    extent:     np.ndarray      # (3,) axis-aligned bounding box extents (L,W,H)
    bbox_min:   np.ndarray      # (3,) AABB minimum corner
    bbox_max:   np.ndarray      # (3,) AABB maximum corner


def _dbscan(pts: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Pure-numpy DBSCAN via a KD-tree (no sklearn dependency).

    :param pts:         ``(P, d)`` float array of points.
    :param eps:         Neighbourhood radius.
    :param min_samples: Minimum neighbours (inclusive of the point itself).
    :returns:           Integer label array of length P. ``-1`` = noise.
    """
    P = pts.shape[0]
    labels = np.full(P, -1, dtype=np.int32)
    visited = np.zeros(P, dtype=bool)
    tree    = cKDTree(pts)

    cluster_id = 0
    for i in range(P):
        if visited[i]:
            continue
        visited[i] = True
        neighbours = tree.query_ball_point(pts[i], eps)
        if len(neighbours) < min_samples:
            continue  # noise (label stays -1)
        labels[i]  = cluster_id
        queue = list(neighbours)
        while queue:
            j = queue.pop()
            if not visited[j]:
                visited[j] = True
                nn_j = tree.query_ball_point(pts[j], eps)
                if len(nn_j) >= min_samples:
                    queue.extend(nn_j)
            if labels[j] == -1:
                labels[j] = cluster_id
        cluster_id += 1

    return labels


def cluster_points(
    sample: StereoSample,
    config: Optional[ClusterConfig] = None,
) -> Tuple[np.ndarray, List[ClusterStats]]:
    """Cluster LiDAR returns with DBSCAN and return per-point labels + stats.

    :param sample: A :class:`~data.StereoSample`.
    :param config: :class:`ClusterConfig`; uses defaults when ``None``.
    :returns: ``(labels, cluster_stats)`` where ``labels`` is a ``(P,)``
              int32 array aligned with the chosen point set, and
              ``cluster_stats`` is a list of :class:`ClusterStats` (sorted by
              descending point count, noise cluster excluded).
    """
    cfg = config or ClusterConfig()

    xyz_full = (
        sample.points_outside_boxes_xyz.astype(np.float32)
        if cfg.background_only
        else sample.lidar_xyz.astype(np.float32)
    )

    if xyz_full.shape[0] == 0:
        return np.zeros(0, dtype=np.int32), []

    # height pre-filter
    z_ok = (xyz_full[:, 2] >= cfg.z_range[0]) & (xyz_full[:, 2] <= cfg.z_range[1])
    xyz  = xyz_full[z_ok]

    if xyz.shape[0] == 0:
        labels_out = np.full(xyz_full.shape[0], -1, dtype=np.int32)
        return labels_out, []

    pts_for_cluster = xyz[:, :2] if cfg.use_bev else xyz
    labels_filtered = _dbscan(pts_for_cluster, cfg.eps_m, cfg.min_samples)

    # map back to original indexing
    labels_out              = np.full(xyz_full.shape[0], -1, dtype=np.int32)
    labels_out[z_ok]        = labels_filtered

    # --- per-cluster statistics ---
    unique_labels = sorted(set(labels_filtered.tolist()) - {-1})
    stats: List[ClusterStats] = []
    for lbl in unique_labels:
        pts_lbl   = xyz[labels_filtered == lbl]
        centroid  = pts_lbl.mean(axis=0)
        bbox_min  = pts_lbl.min(axis=0)
        bbox_max  = pts_lbl.max(axis=0)
        extent    = bbox_max - bbox_min
        stats.append(
            ClusterStats(
                label=lbl,
                num_points=len(pts_lbl),
                centroid=centroid,
                extent=extent,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
            )
        )
    stats.sort(key=lambda s: s.num_points, reverse=True)
    return labels_out, stats



#visualize the BEV feature map
def visualize_bev(bev: torch.Tensor, title: str = "BEV Feature Map"):
    import matplotlib.pyplot as plt
    plt.imshow(bev)
    plt.title(title)
    plt.show()

# Demo helpers
def _lidar_bev(
    sample,
    device: "torch.device",
    x_range=(0.0, 50.0),
    y_range=(-20.0, 20.0),
    z_range=(-3.0, 1.0),
) -> "torch.Tensor":
    """Run the PointPillars branch on one StereoSample.

    Returns
    -------
    bev : (C, nx, ny) float32 tensor on *device*.
    """
    cfg    = PillarConfig(x_range=x_range, y_range=y_range, z_range=z_range)
    branch = PointPillarsBranch(cfg).to(device).eval()
    pts    = sample.lidar_xyz.astype(np.float32)
    inten  = sample.lidar_features["intensity"][:, None].astype(np.float32)
    pts    = np.concatenate([pts, inten], axis=1)
    with torch.no_grad():
        return branch(pts, device=device)


def _camera_bev(
    sample,
    device: "torch.device",
    target_hw=(192, 640),
    x_range=(0.0, 50.0),
    y_range=(-20.0, 20.0),
) -> "torch.Tensor":
    """Run the MonoBEV (Lift-Splat-Shoot) branch on one StereoSample.

    Handles image resizing, K scaling, and extrinsic conversion.

    Returns
    -------
    bev : (C, nx, ny) float32 tensor on *device*.
    """
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
    device: "torch.device",
    target_hw=(192, 640),
    x_range=(0.0, 50.0),
    y_range=(-20.0, 20.0),
) -> "torch.Tensor":
    """Run the :class:`StereoBEVBranch` on one StereoSample.

    Returns
    -------
    bev : (C, nx, ny) float32 tensor on *device*, pixel-aligned with the
          LiDAR and MonoBEV camera BEVs.
    """
    cfg    = StereoBEVConfig(x_range=x_range, y_range=y_range)
    branch = StereoBEVBranch(cfg=cfg, target_hw=target_hw)
    return branch(sample, device)


def _print_bev_stats(name: str, bev: "np.ndarray") -> None:
    """Print min / max / non-zero% for a BEV feature map."""
    nonzero = float((bev != 0).mean()) * 100
    print(f"{name:<12s}  min={bev.min():.4f}  max={bev.max():.4f}"
          f"  non-zero={nonzero:.1f}%")


#: The StereoBEV outputs *learned* feature channels (not geometric ones).
#: Labels shown in the comparison plot are generic feature indices.
def _stereo_ch_label(ch: int) -> str:
    return f"feat ch={ch}"


def _plot_bev_comparison(
    img_np: "np.ndarray",
    bev_camera: "np.ndarray",
    bev_lidar: "np.ndarray",
    title: str,
    bev_stereo: "Optional[np.ndarray]" = None,
    channels=(0, 8, 16, 32),
    stereo_channels=(1, 2, 4, 8),
    save_path: str = "monobev_test_output.png",
) -> None:
    """Camera image + per-branch BEV channels, one row per branch.

    Rows: camera image | camera BEV | LiDAR BEV | stereo BEV (when provided).
    Each panel uses an independent 1st–99th-percentile colour scale so weak
    channels aren't crushed by stronger ones in the same tensor.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    def _panel(ax, feat, label, cmap):
        lo = float(np.percentile(feat,  1))
        hi = float(np.percentile(feat, 99))
        if hi <= lo:
            lo, hi = float(feat.min()), float(feat.max()) + 1e-9
        im = ax.imshow(feat, cmap=cmap, origin="lower", vmin=lo, vmax=hi)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Y (lateral →)")
        ax.set_ylabel("X (forward →)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n_rows = 4 if bev_stereo is not None else 3
    fig = plt.figure(figsize=(18, 3.3 * n_rows))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(n_rows, 4, figure=fig, hspace=0.5, wspace=0.25)

    ax0 = fig.add_subplot(gs[0, :])
    ax0.imshow(img_np)
    ax0.set_title("Left camera image", fontsize=11)
    ax0.axis("off")

    for col, ch in enumerate(channels):
        _panel(fig.add_subplot(gs[1, col]),
               bev_camera[ch], f"Camera BEV ch={ch}", "plasma")

    for col, ch in enumerate(channels):
        ch_lid = min(ch, bev_lidar.shape[0] - 1)
        _panel(fig.add_subplot(gs[2, col]),
               bev_lidar[ch_lid], f"LiDAR BEV ch={ch_lid}", "viridis")

    if bev_stereo is not None:
        for col, ch in enumerate(stereo_channels):
            ch_s = min(ch, bev_stereo.shape[0] - 1)
            _panel(fig.add_subplot(gs[3, col]),
                   bev_stereo[ch_s], f"Stereo BEV {_stereo_ch_label(ch_s)}", "turbo")

    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"\nFigure saved → {save_path}")
    plt.show()


if __name__ == "__main__":
    from data import Py123dDataset

    dataset = Py123dDataset(split_names=["av2-sensor_val"])
    sample  = dataset[0].to_stereo_sample()
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Frame  : {sample.dataset}  log={sample.log_name}  iter={sample.iteration}")
    print(f"Image  : {sample.image_left.shape}   LiDAR : {sample.lidar_xyz.shape[0]} pts")
    print(f"Device : {device}")

    bev_lidar  = _lidar_bev(sample, device)
    bev_camera = _camera_bev(sample, device)
    bev_stereo = _stereo_bev(sample, device)

    bev_lid_np = bev_lidar.detach().cpu().numpy()
    bev_cam_np = bev_camera.detach().cpu().numpy()
    bev_ste_np = bev_stereo.detach().cpu().numpy()

    _print_bev_stats("Camera BEV", bev_cam_np)
    _print_bev_stats("LiDAR  BEV", bev_lid_np)
    _print_bev_stats("Stereo BEV", bev_ste_np)

    _plot_bev_comparison(
        img_np     = sample.image_left,
        bev_camera = bev_cam_np,
        bev_lidar  = bev_lid_np,
        bev_stereo = bev_ste_np,
        title      = f"MonoBEV + PointPillars + StereoBEV (grounded splat)  |  {sample.dataset}  iter={sample.iteration}",
    )

