"""Tests + a runnable comparison for the stereo depth (``data.py``).

Two ways to use this file:

* ``pytest tests/test_stereo.py`` — headless assertions that rectification,
  ``cv2.StereoSGBM`` disparity, metric depth, the ego point cloud and the
  stereo BEV are all well-formed, **plus the key correctness check**: the dense
  stereo depth agrees with the sparse LiDAR depth (the dataset's only depth
  ground truth) to within a tight error budget.

* ``python tests/test_stereo.py`` — prints a stereo-vs-LiDAR depth comparison
  table over a few frames (MAE / median / RMSE / %within-2m) and, with
  ``--save PATH``, writes a disparity / depth / BEV figure.

Nothing below is AV2-specific; pass a different ``--split`` for any py123d
stereo dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections.abc import Sequence

import numpy as np

# Make the repo root importable so ``data`` / ``stereo`` resolve from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import (  # noqa: E402
    Py123dDataset,
    StereoSample,
    StereoSGBMConfig,
    build_rectification,
    disparity_to_depth,
    stereo_bev,
    stereo_depth,
    stereo_point_cloud,
)
from network import PillarConfig  # noqa: E402

DEFAULT_SPLIT = "kitti360_train"


def build_dataset(max_num_scenes: int = 1, split: str = DEFAULT_SPLIT) -> Py123dDataset:
    return Py123dDataset(split_names=[split], max_num_scenes=max_num_scenes)


def compare_depth_to_lidar(
    sample: StereoSample,
    cfg: StereoSGBMConfig | None = None,
    min_m: float = 1.0,
    max_m: float = 60.0,
) -> dict[str, float] | None:
    """Stereo depth vs. the sparse LiDAR depth on their shared pixels.

    Both depths live in the **original left image** frame: ``sample.depth_left``
    is LiDAR projected there, and :func:`stereo_depth` re-projects the dense
    stereo depth into the same frame. We compare only where both are valid and
    inside ``[min_m, max_m]`` (passive stereo is unreliable past ~60 m).

    :returns: a metrics dict, or ``None`` if there is no LiDAR depth / overlap.
    """
    if sample.depth_left is None:
        return None

    sd = stereo_depth(sample, cfg)
    gt, est = sample.depth_left, sd.depth_left
    mask = (gt > min_m) & (gt < max_m) & (est > 0)
    if int(mask.sum()) == 0:
        return None

    err = est[mask] - gt[mask]
    ae = np.abs(err)
    return {
        "n":        float(mask.sum()),
        "mae":      float(ae.mean()),
        "median":   float(np.median(ae)),
        "rmse":     float(np.sqrt((err ** 2).mean())),
        "within_1m": float((ae < 1.0).mean()),
        "within_2m": float((ae < 2.0).mean()),
    }


# --------------------------------------------------------------------------- #
# pytest
# --------------------------------------------------------------------------- #
import pytest  # noqa: E402


@pytest.fixture(scope="module")
def sample() -> StereoSample:
    dataset = build_dataset()
    assert len(dataset) > 0, "dataset produced no frames"
    return dataset[0].to_stereo_sample()


@pytest.fixture(scope="module")
def depth(sample: StereoSample):
    return stereo_depth(sample)


def test_rectification_is_wellformed(sample: StereoSample):
    h, w = sample.image_left.shape[:2]
    rect = build_rectification(sample.calibration, (h, w))
    assert rect.size == (w, h)
    assert rect.Q.shape == (4, 4)
    assert rect.R1.shape == (3, 3)
    assert rect.map_lx.shape == (h, w) and rect.map_ly.shape == (h, w)
    assert rect.fx_rect > 0
    assert 0.0 < rect.baseline_m < 2.0  # KITTI-360 ~0.6 m (AV2 was ~0.5 m)


def test_disparity_to_depth_formula():
    """z = fx·baseline/disp, and disp<=0 → 0 (invalid)."""
    disp = np.array([[0.0, 10.0], [20.0, -1.0]], dtype=np.float32)
    z = disparity_to_depth(disp, fx=1000.0, baseline_m=0.5)
    assert z[0, 0] == 0.0 and z[1, 1] == 0.0          # invalid disparities
    assert np.isclose(z[0, 1], 1000.0 * 0.5 / 10.0)   # 50 m
    assert np.isclose(z[1, 0], 1000.0 * 0.5 / 20.0)   # 25 m


def test_depth_is_dense_and_sane(depth):
    h, w = depth.depth.shape
    valid = depth.depth > 0
    # SGBM should match a non-trivial fraction of the frame ...
    assert valid.mean() > 0.20, f"stereo depth too sparse: {valid.mean():.2%}"
    # ... with all valid depths inside the configured metric range.
    d = depth.depth[valid]
    assert d.min() >= 0.5 - 1e-3 and d.max() <= 80.0 + 1e-3
    assert depth.disparity.shape == (h, w)


def test_point_cloud_in_ego_frame(sample: StereoSample):
    xyz, rgb = stereo_point_cloud(sample)
    assert xyz.ndim == 2 and xyz.shape[1] == 3
    assert rgb.shape == xyz.shape
    assert xyz.shape[0] > 1000, "stereo cloud unexpectedly tiny"
    assert np.isfinite(xyz).all()
    # Camera looks forward (+x in ego): the bulk of points must be in front.
    assert (xyz[:, 0] > 0).mean() > 0.8


def test_bev_shape_matches_lidar_grid(sample: StereoSample):
    """Stereo BEV must be pixel-aligned with the PointPillars LiDAR grid."""
    nx, ny = PillarConfig().grid_size  # (200, 160) with defaults
    bev = stereo_bev(sample)
    assert bev.shape == (7, nx, ny)
    assert bev.dtype == np.float32
    assert np.isfinite(bev).all()
    assert bev[0].max() <= 1.0 and bev[0].min() >= 0.0  # occupancy is binary-ish
    assert bev[0].mean() > 0.02, "stereo BEV essentially empty"


def test_bev_aligns_with_raw_lidar(sample: StereoSample):
    """Binned in the *ego* frame, stereo and LiDAR must occupy the same region.

    A Y-flip or frame error would mirror the cloud and collapse the overlap, so
    a healthy recall of LiDAR cells by stereo is a real geometry check (not just
    a depth-magnitude check, which is done separately against depth_left).
    """
    cfg = PillarConfig()
    nx, ny = cfg.grid_size
    res = cfg.pillar_size

    bev = stereo_bev(sample)
    stereo_occ = bev[0] > 0

    xyz = sample.lidar_xyz
    m = (
        (xyz[:, 0] >= cfg.x_range[0]) & (xyz[:, 0] < cfg.x_range[1]) &
        (xyz[:, 1] >= cfg.y_range[0]) & (xyz[:, 1] < cfg.y_range[1]) &
        (xyz[:, 2] >= cfg.z_range[0]) & (xyz[:, 2] < cfg.z_range[1])
    )
    xyz = xyz[m]
    ix = ((xyz[:, 0] - cfg.x_range[0]) / res).astype(np.int64).clip(0, nx - 1)
    iy = ((xyz[:, 1] - cfg.y_range[0]) / res).astype(np.int64).clip(0, ny - 1)
    lidar_occ = np.zeros((nx, ny), dtype=bool)
    lidar_occ[ix, iy] = True

    recall = (stereo_occ & lidar_occ).sum() / max(1, lidar_occ.sum())
    assert recall > 0.20, f"stereo/LiDAR BEV overlap too low ({recall:.2%}) — frame mismatch?"


def test_depth_matches_lidar_depth(sample: StereoSample):
    """The headline correctness check: stereo depth ≈ LiDAR depth.

    Needs ``sample.depth_left`` (precomputed sparse LiDAR depth); KITTI-360 has
    no precomputed depth maps, so this **skips** unless they were generated.
    Reference figures measured on AV2: median |err| ≈ 0.4 m, ~82 % within 2 m.
    Thresholds are loosened for margin across frames / OpenCV versions.
    """
    metrics = compare_depth_to_lidar(sample)
    if metrics is None:
        pytest.skip("no LiDAR depth (depth_left) or no overlap for this frame")

    assert metrics["n"] > 500, "too few shared pixels to trust the comparison"
    assert metrics["median"] < 1.5, f"median depth error too high: {metrics['median']:.2f} m"
    assert metrics["within_2m"] > 0.60, f"only {metrics['within_2m']:.0%} of pixels within 2 m"


# --------------------------------------------------------------------------- #
# Runnable comparison / visualisation
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare stereo depth against LiDAR depth.")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="py123d split to load")
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 50, 100],
                        help="frame indices to evaluate")
    parser.add_argument("--downscale", type=float, default=1.0, help="SGBM matching scale")
    parser.add_argument("--save", default=None, help="save a disparity/depth/BEV figure for the first frame")
    args = parser.parse_args(argv)

    dataset = build_dataset(split=args.split)
    cfg = StereoSGBMConfig(downscale=args.downscale)
    print(f"{dataset}\n")
    print(f"{'frame':>6} {'N':>7} {'MAE':>7} {'median':>7} {'RMSE':>7} {'<1m':>6} {'<2m':>6}")

    all_ae = []
    for i in args.frames:
        if i >= len(dataset):
            continue
        s = dataset[i].to_stereo_sample()
        mtr = compare_depth_to_lidar(s, cfg)
        if mtr is None:
            print(f"{i:>6}  (no LiDAR depth / overlap)")
            continue
        print(f"{i:>6} {int(mtr['n']):>7} {mtr['mae']:>6.2f}m {mtr['median']:>6.2f}m "
              f"{mtr['rmse']:>6.2f}m {mtr['within_1m']*100:>5.1f}% {mtr['within_2m']*100:>5.1f}%")
        all_ae.append((mtr["mae"], mtr["within_2m"]))

    if all_ae:
        mae = np.mean([a for a, _ in all_ae])
        w2 = np.mean([w for _, w in all_ae])
        print(f"\nmean MAE={mae:.2f} m   mean within-2m={w2*100:.1f}%")

    if args.save:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        s = dataset[args.frames[0]].to_stereo_sample()
        sd = stereo_depth(s, cfg)
        bev = stereo_bev(s, cfg)
        fig, ax = plt.subplots(2, 2, figsize=(16, 11))
        ax[0, 0].imshow(sd.rect_left); ax[0, 0].set_title("rectified left"); ax[0, 0].axis("off")
        d = sd.disparity.copy(); d[d <= 0] = np.nan
        ax[0, 1].imshow(d, cmap="magma"); ax[0, 1].set_title("SGBM disparity (px)"); ax[0, 1].axis("off")
        z = sd.depth.copy(); z[z <= 0] = np.nan
        ax[1, 0].imshow(z, cmap="viridis_r", vmax=60); ax[1, 0].set_title("metric depth (m)"); ax[1, 0].axis("off")
        ax[1, 1].imshow(bev[2].T, origin="lower", cmap="turbo", aspect="auto")
        ax[1, 1].set_title("stereo BEV max-height"); ax[1, 1].set_xlabel("x (fwd)"); ax[1, 1].set_ylabel("y (lat)")
        fig.tight_layout(); fig.savefig(args.save, dpi=90, bbox_inches="tight")
        print(f"Saved figure → {args.save}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
