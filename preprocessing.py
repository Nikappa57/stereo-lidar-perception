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

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from data import Calibration, StereoSample

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
class BEVConfig:
    """Configuration for :func:`bev_map`."""
    x_range:    Tuple[float, float] = (-40.0,  40.0)  # forward (ego X), metres
    y_range:    Tuple[float, float] = (-40.0,  40.0)  # lateral (ego Y), metres
    z_range:    Tuple[float, float] = (-3.0,    1.0)  # height  (ego Z), metres
    resolution: float               = 0.1              # metres per pixel
    #: Channels in the output map (one per sub-list entry):
    #:  0 – max normalised height  [0,1]
    #:  1 – point density (log1p normalised)
    #:  2 – intensity mean (if available, else zeros)
    num_channels: int = 3


def bev_map(
    sample: StereoSample,	
    config: Optional[BEVConfig] = None,
) -> np.ndarray:
    """Build a Bird's-Eye-View feature map from the LiDAR point cloud.

    The ego frame is used directly (X forward, Y left, Z up for AV2).
    Points outside ``config.{x,y,z}_range`` are discarded.

    :param sample: A :class:`~data.StereoSample`.
    :param config: :class:`BEVConfig`; uses defaults when ``None``.
    :returns: Float32 array ``(C, H, W)`` where ``H`` spans the Y range and
              ``W`` spans the X range (both at ``config.resolution`` m/px).
    """
    cfg = config or BEVConfig()
    xyz = sample.lidar_xyz.astype(np.float32)
    if xyz.shape[0] == 0:
        h = int(round((cfg.y_range[1] - cfg.y_range[0]) / cfg.resolution))
        w = int(round((cfg.x_range[1] - cfg.x_range[0]) / cfg.resolution))
        return np.zeros((cfg.num_channels, h, w), dtype=np.float32)

    # --- spatial filter ---
    mask = (
        (xyz[:, 0] >= cfg.x_range[0]) & (xyz[:, 0] < cfg.x_range[1])
        & (xyz[:, 1] >= cfg.y_range[0]) & (xyz[:, 1] < cfg.y_range[1])
        & (xyz[:, 2] >= cfg.z_range[0]) & (xyz[:, 2] < cfg.z_range[1])
    )
    xyz = xyz[mask]

    # --- intensity (optional channel) ---
    intensity: Optional[np.ndarray] = None
    if sample.lidar_features is not None:
        for key in ("intensity", "reflectance", "intensity_lidar"):
            if key in sample.lidar_features:
                intensity = sample.lidar_features[key].astype(np.float32)[mask]
                break

    # --- grid indices ---
    h = int(round((cfg.y_range[1] - cfg.y_range[0]) / cfg.resolution))
    w = int(round((cfg.x_range[1] - cfg.x_range[0]) / cfg.resolution))
    ix = np.floor((xyz[:, 0] - cfg.x_range[0]) / cfg.resolution).astype(np.int32)
    iy = np.floor((xyz[:, 1] - cfg.y_range[0]) / cfg.resolution).astype(np.int32)
    ix = np.clip(ix, 0, w - 1)
    iy = np.clip(iy, 0, h - 1)

    # --- fill channels ---
    z_norm  = ((xyz[:, 2] - cfg.z_range[0]) / (cfg.z_range[1] - cfg.z_range[0])).clip(0.0, 1.0)
    out     = np.zeros((cfg.num_channels, h, w), dtype=np.float32)

    # Channel 0: max height (scatter-reduce)
    np.maximum.at(out[0], (iy, ix), z_norm)

    # Channel 1: density (point count, log1p-normalised)
    count = np.zeros((h, w), dtype=np.float32)
    np.add.at(count, (iy, ix), 1.0)
    max_count = count.max() if count.max() > 0 else 1.0
    out[1] = np.log1p(count) / np.log1p(max_count)

    # Channel 2: mean intensity
    if intensity is not None and cfg.num_channels >= 3:
        intsum = np.zeros((h, w), dtype=np.float32)
        np.add.at(intsum, (iy, ix), intensity)
        valid_px = count > 0
        out[2, valid_px] = intsum[valid_px] / count[valid_px]
        # normalise to [0, 1]
        peak = out[2].max()
        if peak > 0:
            out[2] /= peak

    return out



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
