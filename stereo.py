"""Stereo depth for the stereo-lidar-perception project.

This is the **stereo** counterpart to the LiDAR (PointPillars) and monocular
(MonoBEV / Lift-Splat-Shoot) branches. It turns the raw left/right image pair of
a :class:`~data.StereoSample` into:

* a **disparity** map (``cv2.StereoSGBM``),
* a metric **depth** map (``depth = fx · baseline / disparity``),
* a **3-D point cloud** in the ego frame (so it can be treated exactly like the
  LiDAR cloud), and
* a **Bird's-Eye-View** feature map pixel-aligned with the LiDAR / camera BEV
  grids, ready for fusion.

Why rectify?  The AV2 ``pcam_stereo_l`` / ``pcam_stereo_r`` pair is *almost* but
not exactly rectified (≈0.4° relative rotation, a couple of pixels of vertical
principal-point offset, ~1.4 px focal mismatch). ``cv2.StereoSGBM`` assumes the
epipolar lines are horizontal, so we run ``cv2.stereoRectify`` first; otherwise
the matcher leaks disparity into vertical error and depth is biased.

Quick start::

    from data import Py123dDataset
    from stereo import stereo_depth, stereo_bev

    sample = Py123dDataset(split_names=["av2-sensor_val"])[0].to_stereo_sample()

    sd  = stereo_depth(sample)          # SGBM disparity + metric depth
    bev = stereo_bev(sample)            # (C, nx, ny) aligned with LiDAR BEV
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

import cv2
import numpy as np

from data import Calibration, StereoSample


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class StereoSGBMConfig:
    """All knobs for the SGBM matcher and the disparity→depth conversion.

    The SGBM smoothness penalties follow the OpenCV convention
    ``P = factor · channels · block_size²`` so ``block_size`` can be tuned
    without re-tuning ``P1`` / ``P2`` by hand.
    """

    # ---- disparity search (FULL-res pixels; multiple of 16) ----
    min_disparity:    int = 0
    num_disparities:  int = 256
    block_size:       int = 5      # odd, typically 3..11
    # ---- smoothness (P1 < P2) ----
    p1_factor:        int = 8
    p2_factor:        int = 32
    # ---- post-match consistency / cleanup ----
    uniqueness_ratio:    int = 10
    speckle_window_size: int = 100
    speckle_range:       int = 2
    disp12_max_diff:     int = 1
    pre_filter_cap:      int = 63
    mode:                int = cv2.STEREO_SGBM_MODE_SGBM_3WAY

    # ---- depth conversion / filtering ----
    min_depth_m: float = 0.5
    max_depth_m: float = 80.0

    # ---- speed: matching is done at ``downscale`` of full resolution ----
    #: Rectification is always full-res; only the matching is run on the
    #: downscaled rectified pair, then the disparity is scaled back to full
    #: res, so metric depth is unaffected. ``num_disparities`` is the *full-res*
    #: search range; it is scaled internally for the downscaled matcher.
    #: 0.5 ≈ 4× faster than full 2048×1550. Set 1.0 for best quality.
    downscale: float = 0.5


@dataclass
class StereoBEVConfig:
    """BEV grid for :func:`stereo_bev` — keep aligned with ``PillarConfig``."""

    x_range:   Tuple[float, float] = (0.0,   50.0)   # forward
    y_range:   Tuple[float, float] = (-20.0, 20.0)   # lateral
    z_range:   Tuple[float, float] = (-3.0,   1.0)   # height pre-filter
    bev_res_m: float               = 0.25
    append_rgb: bool               = True            # add mean RGB channels


# --------------------------------------------------------------------------- #
# Rectification
# --------------------------------------------------------------------------- #
@dataclass
class Rectification:
    """Output of :func:`build_rectification` — everything SGBM + lift need.

    All quantities are at **full image resolution**. Down-scaling for speed is
    applied only to the matching step (see :func:`stereo_depth`), never here, so
    ``Q`` / ``fx_rect`` always describe the full-res rectified geometry.
    """

    map_lx: np.ndarray            # left  remap x (for cv2.remap)
    map_ly: np.ndarray            # left  remap y
    map_rx: np.ndarray            # right remap x
    map_ry: np.ndarray            # right remap y
    Q:      np.ndarray            # (4, 4) disparity→3D reprojection matrix
    R1:     np.ndarray            # (3, 3) rotates original-left cam → rectified-left cam
    P1:     np.ndarray            # (3, 4) rectified-left projection matrix
    size:   Tuple[int, int]       # (W, H) of the rectified image
    fx_rect:  float               # rectified focal length (pixels)
    baseline_m: float             # stereo baseline (metres)


def build_rectification(
    calib: Calibration,
    image_hw: Tuple[int, int],
) -> Rectification:
    """Build the full-res stereo rectification maps + reprojection matrix ``Q``.

    :param calib:    Sample :class:`~data.Calibration` (intrinsics + extrinsics).
    :param image_hw: Original ``(H, W)`` of the stereo images.
    """
    h, w = image_hw
    KL = calib.left_intrinsics.astype(np.float64)
    KR = calib.right_intrinsics.astype(np.float64)
    dist = np.zeros(5)  # stereo pair is undistorted (verified in docs/data.md)

    # Relative pose left→right:  X_right = R · X_left + T   (OpenCV convention).
    T_l2r = np.linalg.inv(calib.right_to_ego) @ calib.left_to_ego
    R = T_l2r[:3, :3]
    T = T_l2r[:3, 3]

    R1, R2, P1, P2, Q, _roi1, _roi2 = cv2.stereoRectify(
        KL, dist, KR, dist, (w, h), R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )

    map_lx, map_ly = cv2.initUndistortRectifyMap(KL, dist, R1, P1, (w, h), cv2.CV_32FC1)
    map_rx, map_ry = cv2.initUndistortRectifyMap(KR, dist, R2, P2, (w, h), cv2.CV_32FC1)

    return Rectification(
        map_lx=map_lx, map_ly=map_ly, map_rx=map_rx, map_ry=map_ry,
        Q=Q, R1=R1, P1=P1, size=(w, h),
        fx_rect=float(P1[0, 0]), baseline_m=float(calib.stereo_baseline_m),
    )


# --------------------------------------------------------------------------- #
# Disparity / depth
# --------------------------------------------------------------------------- #
def _build_matcher(cfg: StereoSGBMConfig) -> "cv2.StereoSGBM":
    ch = 1  # matching on grayscale
    return cv2.StereoSGBM_create(
        minDisparity=cfg.min_disparity,
        numDisparities=cfg.num_disparities,
        blockSize=cfg.block_size,
        P1=cfg.p1_factor * ch * cfg.block_size ** 2,
        P2=cfg.p2_factor * ch * cfg.block_size ** 2,
        disp12MaxDiff=cfg.disp12_max_diff,
        uniquenessRatio=cfg.uniqueness_ratio,
        speckleWindowSize=cfg.speckle_window_size,
        speckleRange=cfg.speckle_range,
        preFilterCap=cfg.pre_filter_cap,
        mode=cfg.mode,
    )


def compute_disparity(
    rect_left: np.ndarray,
    rect_right: np.ndarray,
    cfg: Optional[StereoSGBMConfig] = None,
) -> np.ndarray:
    """Run SGBM on an **already rectified** stereo pair.

    :param rect_left:  rectified left image  ``(H, W, 3)`` uint8 or ``(H, W)``.
    :param rect_right: rectified right image, same shape.
    :returns: float32 disparity ``(H, W)`` in pixels; ``<= 0`` marks invalid
        (occluded / unmatched) pixels.
    """
    cfg = cfg or StereoSGBMConfig()
    if rect_left.ndim == 3:
        gl = cv2.cvtColor(rect_left,  cv2.COLOR_RGB2GRAY)
        gr = cv2.cvtColor(rect_right, cv2.COLOR_RGB2GRAY)
    else:
        gl, gr = rect_left, rect_right

    matcher = _build_matcher(cfg)
    disp = matcher.compute(gl, gr).astype(np.float32) / 16.0  # SGBM fixed-point
    disp[disp < cfg.min_disparity + 1e-3] = 0.0
    return disp


def disparity_to_depth(disp: np.ndarray, fx: float, baseline_m: float) -> np.ndarray:
    """Convert disparity (pixels) to metric depth ``z = fx · baseline / disp``.

    Invalid (``disp <= 0``) pixels map to ``0.0``.
    """
    depth = np.zeros_like(disp, dtype=np.float32)
    valid = disp > 0
    depth[valid] = (fx * baseline_m) / disp[valid]
    return depth


# --------------------------------------------------------------------------- #
# Top-level: depth
# --------------------------------------------------------------------------- #
@dataclass
class StereoDepth:
    """Bundle returned by :func:`stereo_depth`."""

    depth:        np.ndarray          # (H', W') metric depth, rectified-left frame, 0=invalid
    disparity:    np.ndarray          # (H', W') disparity, pixels
    rect_left:    np.ndarray          # (H', W', 3) rectified left image (for colour / debug)
    rect:         Rectification       # rectification used (carries Q, R1, sizes)
    depth_left:   np.ndarray          # (H, W) depth re-projected into the ORIGINAL left image


def stereo_depth(
    sample: StereoSample,
    cfg: Optional[StereoSGBMConfig] = None,
) -> StereoDepth:
    """Compute stereo depth for a :class:`~data.StereoSample`.

    Pipeline: rectify → SGBM disparity → metric depth. The headline
    ``depth`` is in the **rectified left** frame (where SGBM works); a
    convenience ``depth_left`` re-projects it back into the **original left
    image** so it lines up with ``sample.depth_left`` (the sparse LiDAR depth)
    for evaluation / fusion.

    :param sample: A :class:`~data.StereoSample` with ``image_left`` / ``image_right``.
    :param cfg:    :class:`StereoSGBMConfig`; defaults when ``None``.
    """
    cfg = cfg or StereoSGBMConfig()
    h, w = sample.image_left.shape[:2]
    rect = build_rectification(sample.calibration, (h, w))

    # ---- remap both views into the (full-res) rectified frame ----
    rect_left  = cv2.remap(sample.image_left,  rect.map_lx, rect.map_ly, cv2.INTER_LINEAR)
    rect_right = cv2.remap(sample.image_right, rect.map_rx, rect.map_ry, cv2.INTER_LINEAR)

    # ---- disparity (matched on the downscaled pair, scaled back to full res) ----
    s = cfg.downscale
    if s < 1.0:
        lo = cv2.resize(rect_left,  None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        ro = cv2.resize(rect_right, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        nd = max(16, int(round(cfg.num_disparities * s / 16)) * 16)
        mcfg = replace(cfg, num_disparities=nd)
        disp_s = compute_disparity(lo, ro, mcfg)
        disp = cv2.resize(disp_s, (w, h), interpolation=cv2.INTER_NEAREST) / s
    else:
        disp = compute_disparity(rect_left, rect_right, cfg)

    # ---- disparity → depth (rectified-left frame) ----
    depth = disparity_to_depth(disp, rect.fx_rect, rect.baseline_m)
    depth[(depth < cfg.min_depth_m) | (depth > cfg.max_depth_m)] = 0.0

    # ---- re-project the dense depth into the ORIGINAL left image ----
    depth_left = _depth_to_original_left(depth, rect, sample.calibration, (h, w), cfg)

    return StereoDepth(
        depth=depth, disparity=disp, rect_left=rect_left, rect=rect,
        depth_left=depth_left,
    )


def _depth_to_original_left(
    depth_rect: np.ndarray,
    rect: Rectification,
    calib: Calibration,
    orig_hw: Tuple[int, int],
    cfg: StereoSGBMConfig,
) -> np.ndarray:
    """Re-project rectified-frame depth into the original (full-res) left image.

    Builds a ``(H, W)`` depth map in the original left image, keeping the
    nearest surface per pixel (z-buffer), so it is directly comparable to
    ``sample.depth_left`` (sparse LiDAR depth in the same frame).
    """
    h, w = orig_hw
    xyz_rect, valid = _reproject_to_3d(depth_rect, rect, cfg)
    if xyz_rect.shape[0] == 0:
        return np.zeros((h, w), dtype=np.float32)

    # rectified-left → original-left camera frame
    xyz_left = (rect.R1.T @ xyz_rect.T).T              # (P, 3)
    z = xyz_left[:, 2]
    front = z > cfg.min_depth_m
    xyz_left, z = xyz_left[front], z[front]

    KL = calib.left_intrinsics.astype(np.float64)
    uv = (KL @ xyz_left.T).T
    u = np.round(uv[:, 0] / z).astype(np.int64)
    v = np.round(uv[:, 1] / z).astype(np.int64)
    inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z = u[inb], v[inb], z[inb]

    out = np.full((h, w), np.inf, dtype=np.float32)
    # nearest-surface z-buffer via sorted scatter (closest written last)
    order = np.argsort(-z)
    out[v[order], u[order]] = z[order]
    out[~np.isfinite(out)] = 0.0
    return out


# --------------------------------------------------------------------------- #
# Top-level: point cloud
# --------------------------------------------------------------------------- #
def _reproject_to_3d(
    depth_rect: np.ndarray,
    rect: Rectification,
    cfg: StereoSGBMConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reproject a rectified depth map to 3-D points in the rectified-left frame.

    :returns: ``(xyz (P, 3), valid_mask (H'*W',))`` where ``xyz`` are the points
        with finite, in-range depth.
    """
    disp = np.zeros_like(depth_rect)
    m = depth_rect > 0
    # invert depth back to disparity for cv2.reprojectImageTo3D (uses Q)
    disp[m] = (rect.fx_rect * rect.baseline_m) / depth_rect[m]
    xyz = cv2.reprojectImageTo3D(disp.astype(np.float32), rect.Q.astype(np.float32))
    xyz = xyz.reshape(-1, 3)
    valid = m.reshape(-1) & np.isfinite(xyz).all(axis=1)
    return xyz[valid], valid


def stereo_point_cloud(
    sample: StereoSample,
    cfg: Optional[StereoSGBMConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Lift the stereo depth into an **ego-frame** coloured point cloud.

    The returned cloud is in the same frame as ``sample.lidar_xyz``, so it can
    be dropped straight into any of the LiDAR consumers (BEV, voxels, frustum).

    :param sample: A :class:`~data.StereoSample`.
    :param cfg:    :class:`StereoSGBMConfig`; defaults when ``None``.
    :returns: ``(xyz_ego (P, 3) float32, rgb (P, 3) uint8)``.
    """
    cfg = cfg or StereoSGBMConfig()
    sd = stereo_depth(sample, cfg)

    xyz_rect, valid = _reproject_to_3d(sd.depth, sd.rect, cfg)
    if xyz_rect.shape[0] == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)

    # rectified-left → original-left → ego
    xyz_left = (sd.rect.R1.T @ xyz_rect.T).T
    cam2ego  = sample.calibration.left_to_ego.astype(np.float64)
    xyz_ego  = (cam2ego[:3, :3] @ xyz_left.T).T + cam2ego[:3, 3]

    rgb = sd.rect_left.reshape(-1, 3)[valid]
    return xyz_ego.astype(np.float32), rgb.astype(np.uint8)


# --------------------------------------------------------------------------- #
# Top-level: BEV
# --------------------------------------------------------------------------- #
def stereo_bev(
    sample: StereoSample,
    cfg: Optional[StereoSGBMConfig] = None,
    bev_cfg: Optional[StereoBEVConfig] = None,
) -> np.ndarray:
    """Build a geometric Bird's-Eye-View from the stereo point cloud.

    The grid is pixel-aligned with the LiDAR (PointPillars) and camera
    (MonoBEV) BEVs — same ``x_range`` / ``y_range`` / resolution — so the three
    maps can be concatenated channel-wise for fusion.

    Channels (``C``)::

        0  occupancy (binary)
        1  log-density
        2  max  height (normalised over z_range)
        3  mean height (normalised over z_range)
        4  mean R   ┐ only when bev_cfg.append_rgb
        5  mean G   │
        6  mean B   ┘

    :returns: ``(C, nx, ny)`` float32 BEV, ``ix`` along x (forward), ``iy``
        along y (lateral) — matching :class:`pointpillars.PointPillarsScatter`.
    """
    bcfg = bev_cfg or StereoBEVConfig()
    xyz, rgb = stereo_point_cloud(sample, cfg)

    nx = int(round((bcfg.x_range[1] - bcfg.x_range[0]) / bcfg.bev_res_m))
    ny = int(round((bcfg.y_range[1] - bcfg.y_range[0]) / bcfg.bev_res_m))
    C  = 7 if bcfg.append_rgb else 4
    bev = np.zeros((C, nx, ny), dtype=np.float32)
    if xyz.shape[0] == 0:
        return bev

    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    m = (
        (x >= bcfg.x_range[0]) & (x < bcfg.x_range[1]) &
        (y >= bcfg.y_range[0]) & (y < bcfg.y_range[1]) &
        (z >= bcfg.z_range[0]) & (z < bcfg.z_range[1])
    )
    x, y, z = x[m], y[m], z[m]
    if x.shape[0] == 0:
        return bev

    ix = ((x - bcfg.x_range[0]) / bcfg.bev_res_m).astype(np.int64).clip(0, nx - 1)
    iy = ((y - bcfg.y_range[0]) / bcfg.bev_res_m).astype(np.int64).clip(0, ny - 1)

    z_norm = ((z - bcfg.z_range[0]) / (bcfg.z_range[1] - bcfg.z_range[0])).clip(0.0, 1.0)

    count = np.zeros((nx, ny), dtype=np.float32)
    np.add.at(count, (ix, iy), 1.0)
    occ = count > 0

    bev[0, occ] = 1.0                                   # occupancy
    bev[1] = np.log1p(count) / np.log1p(count.max())    # log density

    np.maximum.at(bev[2], (ix, iy), z_norm)             # max height
    np.add.at(bev[3], (ix, iy), z_norm)                 # mean height (sum→/count)
    bev[3, occ] /= count[occ]

    if bcfg.append_rgb:
        rgb_m = rgb[m].astype(np.float32) / 255.0
        for k in range(3):
            np.add.at(bev[4 + k], (ix, iy), rgb_m[:, k])
            bev[4 + k][occ] /= count[occ]

    return bev


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from data import Py123dDataset

    sample = Py123dDataset(split_names=["av2-sensor_val"])[0].to_stereo_sample()
    sd  = stereo_depth(sample)
    bev = stereo_bev(sample)

    d = sd.depth[sd.depth > 0]
    print(f"Frame      : {sample.dataset}  iter={sample.iteration}")
    print(f"Rect size  : {sd.rect.size}  fx_rect={sd.rect.fx_rect:.1f}  base={sd.rect.baseline_m:.4f} m")
    print(f"Disparity  : valid={float((sd.disparity > 0).mean())*100:.1f}%")
    print(f"Depth      : valid={float((sd.depth > 0).mean())*100:.1f}%  "
          f"min={d.min():.1f} med={np.median(d):.1f} max={d.max():.1f} m")
    xyz, _ = stereo_point_cloud(sample)
    print(f"Point cloud: {xyz.shape[0]} ego-frame points")
    print(f"Stereo BEV : {bev.shape}  occupancy={float(bev[0].mean())*100:.1f}% cells")
