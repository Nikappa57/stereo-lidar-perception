"""End-to-end training sanity: overfit one frame, decode, recover the GT.

Two ways to use this file:

* ``pytest tests/test_overfit.py`` — headless assertions, three levels of the
  same recipe (TargetEncoder -> model -> CenterPointLoss -> backward ->
  CenterPointDecoder on a single real Argoverse 2 frame):

  - ``test_overfit_lidar_only`` — the cheapest differentiable path
    (:class:`network.LidarOnlyDetector`: pillars -> head).
  - ``test_overfit_fused_pipeline_a`` — the **whole network**
    (:class:`network.PipelineA`: PointPillars + StereoBEV -> ConcatConvFusion
    -> head), asserting the camera branch actually receives gradients.
  - ``test_overfit_fused_pipeline_c`` — Pipeline C with cross-attention fusion,
    asserting gradients flow through the attention and near/far gate layers.

* ``python tests/test_overfit.py`` — overfits Pipeline A and Pipeline C on one
  frame, saves two visual panels and a combined A-vs-C comparison to
  ``docs/img/overfit_fused_output.png`` and
  ``docs/img/overfit_pipeline_c_output.png``.

A healthy loop drives the loss down ~50x in 150 steps and the decoded centres
land on the GT (reference: LiDAR-only 21 -> 0.4, all centres within 6 cm).

Needs the converted dataset (like tests/test_data.py); slow on CPU (~minutes).
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import Py123dDataset  # noqa: E402
from evaluation import CenterPointDecoder  # noqa: E402
from network import (LidarOnlyDetector, PipelineA, PipelineC,  # noqa: E402
                     lidar_points)
from train import (TargetEncoder, encode_sample,  # noqa: E402
                   overfit_one_frame)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_sample():
    """Load a single frame from the AV2 validation split.

    If the dataset directory is missing or contains no scenes, raise a clear
    ``RuntimeError`` with instructions. This prevents obscure ``IndexError``
    crashes when users run the visualisation script on a fresh checkout.
    """
    dataset = Py123dDataset(split_names=["av2-sensor_val"], max_num_scenes=1)
    if not getattr(dataset, "scenes", None):
        raise RuntimeError(
            "No scenes found in the AV2 dataset. Please download the Argoverse 2 "
            "validation split and set the appropriate environment variable or "
            "configure ``data.py`` to point at the dataset location."
        )
    # Grab a frame a few iterations after the start to ensure we have enough history.
    frame = dataset.get_frame(
        0, dataset.scenes[0].number_of_history_iterations + 13)
    return frame.to_stereo_sample()


def _assert_recovers_gt(model, inputs, tgt_hm, tgt_off, history):
    """Loss must collapse and every GT centre must decode back within 0.5 m."""
    assert history[-1] < history[0] * 0.2, (
        f"loss did not drop enough: {history[0]:.2f} -> {history[-1]:.2f}")

    model.eval()
    with torch.no_grad():
        pred = model(inputs, device=DEVICE)
    decoder = CenterPointDecoder(score_threshold=0.3)
    det = decoder(pred["heatmap"].cpu(), pred["offset"].cpu())[0]
    # GT centres through the same decode path (targets scaled to logits-like)
    gt = decoder(tgt_hm * 20 - 10, tgt_off)[0]
    assert len(det["boxes_2d"]) > 0, "no detections decoded after overfit"

    dist = torch.cdist(gt["boxes_2d"], det["boxes_2d"])  # (n_gt, n_det)
    min_dist, nearest = dist.min(dim=1)
    recovered = min_dist < 0.5  # metres — the tightest AV2 distance-AP band
    assert recovered.float().mean() > 0.9, (
        f"only {int(recovered.sum())}/{len(recovered)} GT centres within 0.5 m")
    cls_ok = det["classes"][nearest] == gt["classes"]
    assert cls_ok.float().mean() > 0.9, "decoded classes do not match GT"

    print(f"\nloss {history[0]:.2f} -> {history[-1]:.2f}; "
          f"{int(recovered.sum())}/{len(recovered)} GT centres recovered "
          f"(mean {min_dist.mean():.3f} m, max {min_dist.max():.3f} m) on {DEVICE}")


def test_overfit_lidar_only():
    sample = _load_sample()
    tgt_hm, tgt_off = encode_sample(sample, TargetEncoder())
    assert int(tgt_hm.eq(1).sum()) > 0, "no positive centre cells in this frame"

    pts = lidar_points(sample)
    model = LidarOnlyDetector()
    history = overfit_one_frame(model, pts, tgt_hm, tgt_off, steps=150,
                                lr=1e-3, device=DEVICE)
    _assert_recovers_gt(model, pts, tgt_hm, tgt_off, history)


def test_overfit_fused_pipeline_a():
    sample = _load_sample()
    tgt_hm, tgt_off = encode_sample(sample, TargetEncoder())
    assert int(tgt_hm.eq(1).sum()) > 0, "no positive centre cells in this frame"

    model = PipelineA()
    history = overfit_one_frame(model, sample, tgt_hm, tgt_off, steps=150,
                                lr=1e-3, device=DEVICE)

    # the camera branch must have trained, not just the LiDAR path — this is
    # what the StereoBEVBranch eval()/no_grad() removal bought us
    cam_grads = [p.grad for p in model.camera_branch.parameters()
                 if p.grad is not None]
    assert cam_grads, "camera branch received no gradients"
    assert any(float(g.abs().sum()) > 0 for g in cam_grads), (
        "camera branch gradients are all zero")

    _assert_recovers_gt(model, sample, tgt_hm, tgt_off, history)


def test_overfit_fused_pipeline_c():
    """Pipeline C (cross-attention) must converge and propagate grads through attn + gate."""
    sample = _load_sample()
    tgt_hm, tgt_off = encode_sample(sample, TargetEncoder())
    assert int(tgt_hm.eq(1).sum()) > 0, "no positive centre cells in this frame"

    model = PipelineC()
    history = overfit_one_frame(model, sample, tgt_hm, tgt_off, steps=150,
                                lr=1e-3, device=DEVICE)

    # Camera branch must have trained (same as Pipeline A)
    cam_grads = [p.grad for p in model.camera_branch.parameters()
                 if p.grad is not None]
    assert cam_grads, "camera branch received no gradients in Pipeline C"
    assert any(float(g.abs().sum()) > 0 for g in cam_grads), (
        "camera branch gradients are all zero in Pipeline C")

    # Attention projections (q_proj / k_proj / v_proj) must have gradients
    attn = model.detector.fusion.cross_attn
    attn_grad_params = [p for p in attn.parameters() if p.grad is not None
                        and float(p.grad.abs().sum()) > 0]
    assert attn_grad_params, (
        "cross-attention projections received no non-zero gradients")

    # Near/far gate must have gradients
    gate_grads = [p.grad for p in model.detector.fusion.gate.parameters()
                  if p.grad is not None and float(p.grad.abs().sum()) > 0]
    assert gate_grads, "near/far gate received no non-zero gradients"

    _assert_recovers_gt(model, sample, tgt_hm, tgt_off, history)


# --------------------------------------------------------------------------- #
# Visual: overfit the fused Pipeline A on one frame and save a result panel
# --------------------------------------------------------------------------- #
def save_overfit_figure(save_path: str = "docs/img/overfit_fused_output.png",
                        steps: int = 150) -> str:
    """Overfit :class:`network.PipelineA` on one frame and plot the result.

    Panels (all BEV panels in the ego frame, Y lateral / X forward):
    left RGB | encoded target heatmap | learned heatmap after ``steps`` |
    loss curve | decoded detections vs GT centres over the LiDAR density BEV.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    import globals as G

    sample = _load_sample()
    tgt_hm, tgt_off = encode_sample(sample, TargetEncoder())

    model = PipelineA()
    history = overfit_one_frame(model, sample, tgt_hm, tgt_off, steps=steps,
                                lr=1e-3, device=DEVICE)

    model.eval()
    with torch.no_grad():
        pred = model(sample, device=DEVICE)
    decoder = CenterPointDecoder(score_threshold=0.3)
    det = decoder(pred["heatmap"].cpu(), pred["offset"].cpu())[0]
    gt = decoder(tgt_hm * 20 - 10, tgt_off)[0]
    det_xy, gt_xy = det["boxes_2d"].numpy(), gt["boxes_2d"].numpy()

    (x_min, x_max), (y_min, y_max) = G.X_RANGE, G.Y_RANGE
    extent = [y_min, y_max, x_min, x_max]
    nx, ny = G.GRID_SIZE

    # LiDAR density background (same recipe as utils.visualize_bev)
    grid = np.zeros((nx, ny), np.float32)
    pts = sample.lidar_xyz
    m = ((pts[:, 0] >= x_min) & (pts[:, 0] < x_max) &
         (pts[:, 1] >= y_min) & (pts[:, 1] < y_max))
    p = pts[m]
    np.add.at(grid, (((p[:, 0] - x_min) / G.BEV_RES_M).astype(int),
                     ((p[:, 1] - y_min) / G.BEV_RES_M).astype(int)), 1.0)

    def _bev(ax, feat, title, cmap):
        im = ax.imshow(feat, origin="lower", cmap=cmap, extent=extent,
                       aspect="auto")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Y lateral (m)")
        ax.set_ylabel("X forward (m)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig, ax = plt.subplots(1, 5, figsize=(26, 5.5))
    fig.suptitle(
        f"Fused Pipeline A overfit ({steps} steps, one frame) | "
        f"{sample.dataset} log={sample.log_name} iter={sample.iteration} | "
        f"loss {history[0]:.1f} -> {history[-1]:.2f}",
        fontsize=13, fontweight="bold")

    ax[0].imshow(sample.image_left)
    ax[0].set_title("Left RGB", fontsize=10)
    ax[0].axis("off")

    _bev(ax[1], tgt_hm[0].amax(0).numpy(), "Target heatmap (max class)", "hot")
    _bev(ax[2], pred["heatmap"].sigmoid().amax(1)[0].detach().cpu().numpy(),
         "Learned heatmap (max class, σ)", "hot")

    ax[3].plot(history)
    ax[3].set_yscale("log")
    ax[3].set_title("Total loss (log scale)", fontsize=10)
    ax[3].set_xlabel("step")
    ax[3].grid(alpha=0.3)

    _bev(ax[4], np.log1p(grid), "Decoded vs GT (LiDAR density)", "bone")
    if len(gt_xy):
        ax[4].scatter(gt_xy[:, 1], gt_xy[:, 0], s=95, facecolors="none",
                      edgecolors="cyan", lw=1.5, label="GT centre")
    if len(det_xy):
        ax[4].scatter(det_xy[:, 1], det_xy[:, 0], s=30, c="red", marker="x",
                      lw=1.5, label=f"decoded ({len(det_xy)})")
    ax[4].legend(loc="upper right", fontsize=8)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"loss {history[0]:.2f} -> {history[-1]:.2f} | "
          f"decoded {len(det_xy)} vs {len(gt_xy)} GT")
    print(f"Figure saved → {save_path}")
    return save_path


# --------------------------------------------------------------------------- #
# Visual: overfit Pipeline C and save a comparison panel A vs C
# --------------------------------------------------------------------------- #
def save_overfit_figure_pipeline_c(
        save_path: str = "docs/img/overfit_pipeline_c_output.png",
        steps: int = 150) -> str:
    """Overfit Pipeline A and Pipeline C on one frame and compare them.

    Panels (all BEV in ego frame):
    left RGB | target heatmap | Pipeline A heatmap (concat+conv) |
    Pipeline C heatmap (cross-attn) | learned near/far gate | A vs C loss curves.

    The gate panel is the main novelty visualisation: near cells (stereo-dense,
    low gate) appear dark; far cells (LiDAR-reliable, high gate) appear bright.
    Untrained networks give an uninformative gate; after overfitting, the near/far
    structure emerges from the data.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    import globals as G
    from utils import record_activations  # noqa: F401  # available for manual hook debugging

    sample = _load_sample()
    tgt_hm, tgt_off = encode_sample(sample, TargetEncoder())

    # --- overfit both pipelines on the same frame ---
    model_a = PipelineA()
    history_a = overfit_one_frame(model_a, sample, tgt_hm, tgt_off,
                                  steps=steps, lr=1e-3, device=DEVICE)

    model_c = PipelineC()
    history_c = overfit_one_frame(model_c, sample, tgt_hm, tgt_off,
                                  steps=steps, lr=1e-3, device=DEVICE)

    # --- final predictions ---
    decoder = CenterPointDecoder(score_threshold=0.3)
    (x_min, x_max), (y_min, y_max) = G.X_RANGE, G.Y_RANGE
    extent = [y_min, y_max, x_min, x_max]

    model_a.eval()
    with torch.no_grad():
        pred_a = model_a(sample, device=DEVICE)
    hm_a = pred_a["heatmap"].sigmoid().amax(1)[0].cpu().numpy()
    det_a = decoder(pred_a["heatmap"].cpu(), pred_a["offset"].cpu())[0]

    model_c.eval()
    # Capture the gate values during the forward pass
    gate_map = {}
    def _grab_gate(m, inp, out):
        # gate is (B, 1, nx, ny) — squeeze batch and channel
        gate_map["gate"] = out[0, 0].detach().cpu().numpy()
    hook_handle = model_c.detector.fusion.gate.register_forward_hook(_grab_gate)
    with torch.no_grad():
        pred_c = model_c(sample, device=DEVICE)
    hook_handle.remove()
    hm_c = pred_c["heatmap"].sigmoid().amax(1)[0].cpu().numpy()
    det_c = decoder(pred_c["heatmap"].cpu(), pred_c["offset"].cpu())[0]
    gate = gate_map.get("gate", np.zeros((1, 1)))  # fallback if hook missed

    # GT centres in-grid
    centres = sample.boxes_3d_ego[:, :2] if len(sample.boxes_3d_ego) else np.zeros((0, 2))
    m = ((centres[:, 0] >= x_min) & (centres[:, 0] < x_max) &
         (centres[:, 1] >= y_min) & (centres[:, 1] < y_max))
    centres = centres[m]

    # LiDAR density BEV background (shared between the two scatter panels)
    nx, ny = G.GRID_SIZE
    lidar_grid = np.zeros((nx, ny), np.float32)
    pts = sample.lidar_xyz
    pm = ((pts[:, 0] >= x_min) & (pts[:, 0] < x_max) &
          (pts[:, 1] >= y_min) & (pts[:, 1] < y_max))
    p = pts[pm]
    np.add.at(lidar_grid,
              (((p[:, 0] - x_min) / G.BEV_RES_M).astype(int),
               ((p[:, 1] - y_min) / G.BEV_RES_M).astype(int)), 1.0)
    lidar_log = np.log1p(lidar_grid)

    # Decoded box centres for scatter panels
    gt_xy = centres                                      # (N, 2) ego x/y
    det_a_xy = det_a["boxes_2d"].numpy()                 # (M, 2)
    det_c_xy = det_c["boxes_2d"].numpy()                 # (M, 2)

    def _bev(ax, feat, title, cmap, vmin=None, vmax=None):
        im = ax.imshow(feat, origin="lower", cmap=cmap, extent=extent,
                       aspect="auto", vmin=vmin, vmax=vmax)
        if len(centres):
            ax.scatter(centres[:, 1], centres[:, 0], s=40,
                       facecolors="none", edgecolors="cyan", lw=1.2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Y lateral (m)")
        ax.set_ylabel("X forward (m)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def _scatter_panel(ax, det_xy, title):
        """LiDAR density BEV with GT circles and decoded × markers."""
        ax.imshow(lidar_log, origin="lower", cmap="bone", extent=extent,
                  aspect="auto")
        if len(gt_xy):
            ax.scatter(gt_xy[:, 1], gt_xy[:, 0], s=100,
                       facecolors="none", edgecolors="cyan", lw=1.8,
                       label="GT centre", zorder=3)
        if len(det_xy):
            ax.scatter(det_xy[:, 1], det_xy[:, 0], s=60,
                       c="red", marker="x", lw=1.8,
                       label=f"decoded ({len(det_xy)})", zorder=4)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Y lateral (m)")
        ax.set_ylabel("X forward (m)")
        ax.legend(loc="upper right", fontsize=8)

    fig, ax = plt.subplots(2, 4, figsize=(30, 12))
    fig.suptitle(
        f"Pipeline A vs C overfit ({steps} steps) | "
        f"{sample.dataset} iter={sample.iteration} | "
        f"A: {history_a[0]:.1f}->{history_a[-1]:.2f}  "
        f"C: {history_c[0]:.1f}->{history_c[-1]:.2f}",
        fontsize=13, fontweight="bold")

    # Row 0: Left RGB | Target heatmap | Pipeline A heatmap | GT vs Pipeline A
    ax[0, 0].imshow(sample.image_left)
    ax[0, 0].set_title("Left RGB", fontsize=10)
    ax[0, 0].axis("off")

    _bev(ax[0, 1], tgt_hm[0].amax(0).numpy(),
         "Target heatmap (max class)", "hot")

    _bev(ax[0, 2], hm_a,
         f"Pipeline A — concat+conv (loss {history_a[-1]:.3f})", "hot")

    _scatter_panel(ax[0, 3], det_a_xy,
                   f"GT vs Pipeline A\n(cyan=GT, red×=decoded {len(det_a_xy)})")

    # Row 1: Pipeline C heatmap | Near/far gate | Loss curves | GT vs Pipeline C
    _bev(ax[1, 0], hm_c,
         f"Pipeline C — cross-attn (loss {history_c[-1]:.3f})", "hot")

    _bev(ax[1, 1], gate,
         "Learned near/far gate g\n(dark=trust stereo, bright=trust LiDAR)",
         "RdYlGn_r", vmin=0.0, vmax=1.0)

    ax[1, 2].plot(history_a, label="Pipeline A (concat+conv)", color="steelblue")
    ax[1, 2].plot(history_c, label="Pipeline C (cross-attn)", color="darkorange")
    ax[1, 2].set_yscale("log")
    ax[1, 2].set_title("Loss curves (log scale)", fontsize=10)
    ax[1, 2].set_xlabel("step")
    ax[1, 2].legend(fontsize=9)
    ax[1, 2].grid(alpha=0.3)

    _scatter_panel(ax[1, 3], det_c_xy,
                   f"GT vs Pipeline C\n(cyan=GT, red×=decoded {len(det_c_xy)})")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Pipeline A loss {history_a[0]:.2f} -> {history_a[-1]:.2f} | "
          f"decoded {len(det_a['boxes_2d'])} objs")
    print(f"Pipeline C loss {history_c[0]:.2f} -> {history_c[-1]:.2f} | "
          f"decoded {len(det_c['boxes_2d'])} objs")
    print(f"Figure saved -> {save_path}")
    return save_path


if __name__ == "__main__":
    try:
        save_overfit_figure()
        save_overfit_figure_pipeline_c()
    except RuntimeError as e:
        # Friendly message – visualisation skipped when data unavailable.
        print(f"[visualisation skipped] {e}")
