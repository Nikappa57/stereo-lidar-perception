"""Tests and a visual sanity-check for the BEV fusion + head (``network.py``).

Two ways to use this file:

* ``pytest tests/test_network.py`` -- headless assertions that the detector runs
  and produces the right output shapes, and that the fusion shape-guard rejects
  misaligned BEV maps (design doc SS02). These use random tensors, so they are
  fast and need no dataset. Pipeline C-specific tests verify gradient flow
  through the cross-attention and the near/far gate.

* ``python tests/test_network.py`` -- loads one real frame, runs the
  two Stage A branches (LiDAR PointPillars + grounded StereoBEV), fuses them with
  ``BEVDetector`` and saves a figure of the **fused BEV** (plus the two input
  BEVs and the head's centre heatmap) to ``docs/img/bev_fusion_test_output.png``.
  The branches are untrained here -- the point is to see the data flow end-to-end
  and that the fused map is grid-aligned with the ego-frame GT box centres.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Make the repo root importable so ``network`` / ``data`` resolve from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from network import (  # noqa: E402
    CAMERA_BEV_CHANNELS,
    LIDAR_BEV_CHANNELS,
    BEVDetector,
    BEVFusionConfig,
    CenterPointHead,
    CrossAttentionFusion,
    PipelineC,
)

GRID = (200, 160)  # (nx, ny) for x in [0,50] m, y in [-20,20] m @ 0.25 m/cell


# --------------------------------------------------------------------------- #
# pytest: shape + alignment-guard checks (no dataset, no heavy models)
# --------------------------------------------------------------------------- #
def test_detector_output_shapes():
    """Fusing two aligned BEV maps yields heatmap + offset on the same grid."""
    nx, ny = GRID
    bev_camera = torch.randn(CAMERA_BEV_CHANNELS, nx, ny)
    bev_lidar  = torch.randn(LIDAR_BEV_CHANNELS, nx, ny)

    detector = BEVDetector.from_bev_maps(bev_camera, bev_lidar, num_classes=3).eval()
    with torch.no_grad():
        out = detector(bev_camera, bev_lidar)

    assert out["heatmap"].shape == (1, 3, nx, ny)
    assert out["offset"].shape  == (1, 2, nx, ny)


def test_fusion_guard_rejects_misaligned_grids():
    """The SS02 guard fires when the two BEV maps do not share the grid."""
    detector   = BEVDetector(BEVFusionConfig())          # defaults: 64/128 ch, 200x160
    bev_camera = torch.randn(CAMERA_BEV_CHANNELS, *GRID)
    bev_lidar  = torch.randn(LIDAR_BEV_CHANNELS, 100, 80)  # wrong grid
    with pytest.raises(AssertionError):
        detector(bev_camera, bev_lidar)


def test_from_bev_maps_reads_contract():
    """``from_bev_maps`` derives the contract from the actual Stage A tensors."""
    bev_camera = torch.randn(48, 120, 96)   # non-default channels / grid
    bev_lidar  = torch.randn(96, 120, 96)
    detector   = BEVDetector.from_bev_maps(bev_camera, bev_lidar, num_classes=2)
    assert detector.cfg.camera_channels == 48
    assert detector.cfg.lidar_channels  == 96
    assert detector.cfg.grid_size       == (120, 96)
    with torch.no_grad():
        out = detector(bev_camera, bev_lidar)
    assert out["heatmap"].shape == (1, 2, 120, 96)


# --------------------------------------------------------------------------- #
# Pipeline C -- fast headless tests (random tensors, no dataset needed)
# --------------------------------------------------------------------------- #
def test_cross_attention_fusion_output_shape():
    """CrossAttentionFusion honours the BEVFusionConfig channel + grid contract."""
    nx, ny = GRID
    cfg    = BEVFusionConfig()          # 64 cam / 128 lid / 128 out, 200x160
    fusion = CrossAttentionFusion(cfg).eval()

    bev_cam = torch.randn(2, CAMERA_BEV_CHANNELS, nx, ny)
    bev_lid = torch.randn(2, LIDAR_BEV_CHANNELS,  nx, ny)
    with torch.no_grad():
        out = fusion(bev_cam, bev_lid)

    assert out.shape == (2, cfg.out_channels, nx, ny), (
        f"unexpected fused shape {tuple(out.shape)}")


def test_cross_attention_fusion_non_default_window():
    """CrossAttentionFusion works when the grid is not an exact multiple of win."""
    # 13x11 is deliberately awkward (not a multiple of 8)
    cfg    = BEVFusionConfig(camera_channels=32, lidar_channels=64,
                             out_channels=32, grid_size=(13, 11))
    fusion = CrossAttentionFusion(cfg, num_heads=4, win_h=8, win_w=8).eval()
    bev_cam = torch.randn(1, 32, 13, 11)
    bev_lid = torch.randn(1, 64, 13, 11)
    with torch.no_grad():
        out = fusion(bev_cam, bev_lid)
    assert out.shape == (1, 32, 13, 11)


def test_pipeline_c_output_shapes():
    """PipelineC composes correctly: detector carries CrossAttentionFusion,
    and head output shapes are identical to Pipeline A."""
    import globals as G

    nx, ny  = G.GRID_SIZE
    bev_cam = torch.randn(1, CAMERA_BEV_CHANNELS, nx, ny)
    bev_lid = torch.randn(1, LIDAR_BEV_CHANNELS,  nx, ny)

    det = BEVDetector.from_bev_maps(
        bev_cam, bev_lid,
        num_classes=G.NUM_CLASSES,
        fusion_cls=CrossAttentionFusion,
    ).eval()
    with torch.no_grad():
        out = det(bev_cam, bev_lid)

    assert isinstance(det.fusion, CrossAttentionFusion)
    assert out["heatmap"].shape == (1, G.NUM_CLASSES, nx, ny)
    assert out["offset"].shape  == (1, 2, nx, ny)


def test_cross_attention_gradients_flow():
    """Backprop reaches q_proj, k_proj, v_proj and the near/far gate."""
    import globals as G

    nx, ny = 40, 32   # small grid for speed
    cfg = BEVFusionConfig(
        camera_channels=CAMERA_BEV_CHANNELS,
        lidar_channels=LIDAR_BEV_CHANNELS,
        out_channels=32,
        grid_size=(nx, ny),
        num_classes=G.NUM_CLASSES,
    )
    fusion = CrossAttentionFusion(cfg, num_heads=4)
    head   = CenterPointHead(cfg.out_channels, G.NUM_CLASSES)

    bev_cam = torch.randn(1, CAMERA_BEV_CHANNELS, nx, ny)
    bev_lid = torch.randn(1, LIDAR_BEV_CHANNELS,  nx, ny)

    fused = fusion(bev_cam, bev_lid)
    out   = head(fused)
    loss  = out["heatmap"].sigmoid().sum()   # trivial scalar
    loss.backward()

    # attention projection layers must have non-zero gradients
    attn = fusion.cross_attn
    for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        param = getattr(attn, name).weight
        assert param.grad is not None and param.grad.abs().sum() > 0, (
            f"cross_attn.{name}.weight has no gradient")

    # near/far gate must have non-zero gradients
    for i, p in enumerate(fusion.gate.parameters()):
        if p.requires_grad:
            assert p.grad is not None and p.grad.abs().sum() > 0, (
                f"gate param [{i}] has no gradient")


# --------------------------------------------------------------------------- #
# Visual: fuse one real frame and save an image of the fused BEV
# --------------------------------------------------------------------------- #
def _aggregate(bev: torch.Tensor) -> np.ndarray:
    """Collapse a ``(C, nx, ny)`` BEV map to a single ``(nx, ny)`` activation."""
    return bev.detach().cpu().abs().mean(dim=0).numpy()


def _panel(ax, feat, title, cmap, x_range, y_range, centres=None):
    import matplotlib.pyplot as plt

    lo, hi = float(np.percentile(feat, 1)), float(np.percentile(feat, 99))
    if hi <= lo:
        lo, hi = float(feat.min()), float(feat.max()) + 1e-9
    im = ax.imshow(
        feat, origin="lower", cmap=cmap, vmin=lo, vmax=hi, aspect="auto",
        extent=[y_range[0], y_range[1], x_range[0], x_range[1]],
    )
    if centres is not None and len(centres):
        ax.scatter(centres[:, 1], centres[:, 0], s=14, facecolors="none",
                   edgecolors="red", linewidths=1.0, label="GT centres (in grid)")
        ax.legend(loc="upper right", fontsize=7)
    # Lock the view to the BEV extent so out-of-grid scatter cannot rescale it.
    ax.set_xlim(y_range)
    ax.set_ylim(x_range)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Y lateral (m)")
    ax.set_ylabel("X forward (m)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def save_fusion_figure(save_path: str = "docs/img/bev_fusion_test_output.png") -> str:
    """Run both branches + fusion on one frame and save the fused-BEV figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from data import Py123dDataset
    from network import _lidar_bev, _stereo_bev

    dataset = Py123dDataset(split_names=["av2-sensor_val"], max_num_scenes=1)
    frame   = dataset.get_frame(0, dataset.scenes[0].number_of_history_iterations + 13)
    sample  = frame.to_stereo_sample()
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Stage A: the two grid-aligned BEV maps (untrained branches).
    bev_lidar  = _lidar_bev(sample, device)    # (128, nx, ny)
    bev_camera = _stereo_bev(sample, device)   # (64,  nx, ny)

    # BEV fusion + head, built from the actual Stage A outputs.
    detector = BEVDetector.from_bev_maps(
        bev_camera, bev_lidar, num_classes=3).to(device).eval()
    with torch.no_grad():
        fused   = detector.fusion(bev_camera, bev_lidar)   # (1, 128, nx, ny)
        out     = detector.head(fused)
    heatmap = out["heatmap"].sigmoid().amax(dim=1)[0].detach().cpu().numpy()

    x_range, y_range = (0.0, 50.0), (-20.0, 20.0)
    centres = (sample.boxes_3d_ego[:, :2]
               if len(sample.boxes_3d_ego) else np.zeros((0, 2)))
    in_grid = (
        (centres[:, 0] >= x_range[0]) & (centres[:, 0] < x_range[1])
        & (centres[:, 1] >= y_range[0]) & (centres[:, 1] < y_range[1])
    )
    centres = centres[in_grid]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle(
        f"BEV fusion (untrained) | {sample.dataset} "
        f"log={sample.log_name} iter={sample.iteration}",
        fontsize=13, fontweight="bold",
    )
    _panel(axes[0], _aggregate(bev_camera), "Camera BEV (64 ch, |mean|)",
           "plasma",  x_range, y_range)
    _panel(axes[1], _aggregate(bev_lidar),  "LiDAR BEV (128 ch, |mean|)",
           "viridis", x_range, y_range)
    _panel(axes[2], _aggregate(fused[0]),   "Fused BEV (128 ch, |mean|)",
           "magma",   x_range, y_range, centres)
    _panel(axes[3], heatmap,                "Head heatmap (max class, sigma)",
           "inferno", x_range, y_range, centres)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved -> {save_path}")
    return save_path


if __name__ == "__main__":
    save_fusion_figure()
