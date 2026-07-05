# Visualize preprocessing results + network debugging helpers

from contextlib import contextmanager

import numpy as np
import matplotlib.pyplot as plt

from data import StereoSample, frustum_points, voxel_grid, cluster_points
from network import PillarConfig


# --------------------------------------------------------------------------- #
# Network debugging — capture and plot intermediate outputs
# --------------------------------------------------------------------------- #
@contextmanager
def record_activations(model, layer_names):
    """Capture submodule outputs by dotted path during forward passes.

    Registers forward hooks on each named submodule (``model.get_submodule``
    path, e.g. ``"camera_branch.model.context_head"``) and stores a detached
    CPU copy of its output under that name; hooks are removed on exit, so the
    model is untouched afterwards. Works on any ``nn.Module``. Usage::

        pipe = PipelineA().eval()
        with record_activations(pipe, [
                "camera_branch.model.backbone",      # (1, 256, H/8, W/8)
                "camera_branch.model.context_head",  # (1, 64, H/8, W/8)
                "lidar_branch.pfn",                  # (P, 64) pillar features
                "detector.fusion",                   # (1, 128, nx, ny)
        ]) as acts:
            with torch.no_grad():
                pipe(sample, device=device)
        acts["detector.fusion"].shape

    Print ``dict(model.named_modules()).keys()`` to discover valid paths.
    """
    acts, handles = {}, []

    def _make_hook(name):
        def _hook(module, args, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            if isinstance(out, dict):  # e.g. CenterPointHead returns a dict
                acts[name] = {k: v.detach().cpu() for k, v in out.items()}
            else:
                acts[name] = out.detach().cpu()
        return _hook

    for name in layer_names:
        module = model.get_submodule(name)
        handles.append(module.register_forward_hook(_make_hook(name)))
    try:
        yield acts
    finally:
        for h in handles:
            h.remove()


def visualize_pipeline_debug(pipeline, sample, device=None,
                             save_path: str | None = None):
    """One debug forward of a :class:`network.Pipeline` → panel of every stage.

    Panels (BEV ones on the shared ego grid, GT centres overlaid): left RGB |
    camera BEV | LiDAR BEV | fused BEV (channel-collapsed to |mean|) | head
    heatmap (sigmoid, max class) | offset magnitude. Untrained weights give
    structured-but-meaningless maps — the point is checking data flow, grid
    alignment and where activations live, not detection quality.

    Pass ``save_path`` to write a PNG instead of showing interactively.
    """
    import torch

    import globals as G

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.debug = True
    pipeline.to(device).eval()
    with torch.no_grad():
        pipeline(sample, device=device)
    inter = pipeline.intermediates

    (x_min, x_max), (y_min, y_max) = G.X_RANGE, G.Y_RANGE
    extent = [y_min, y_max, x_min, x_max]

    # in-grid GT centres, for the alignment check
    centres = sample.boxes_3d_ego[:, :2] if len(sample.boxes_3d_ego) else np.zeros((0, 2))
    m = ((centres[:, 0] >= x_min) & (centres[:, 0] < x_max) &
         (centres[:, 1] >= y_min) & (centres[:, 1] < y_max))
    centres = centres[m]

    def _panel(ax, feat, title, cmap):
        lo, hi = float(np.percentile(feat, 1)), float(np.percentile(feat, 99))
        if hi <= lo:
            lo, hi = float(feat.min()), float(feat.max()) + 1e-9
        im = ax.imshow(feat, origin="lower", cmap=cmap, vmin=lo, vmax=hi,
                       extent=extent, aspect="auto")
        if len(centres):
            ax.scatter(centres[:, 1], centres[:, 0], s=40, facecolors="none",
                       edgecolors="cyan", lw=1.0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Y lateral (m)")
        ax.set_ylabel("X forward (m)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig, ax = plt.subplots(2, 3, figsize=(17, 11))
    fig.suptitle(f"{type(pipeline).__name__} debug forward | "
                 f"{sample.dataset} iter={sample.iteration} | GT rings = "
                 f"ego-frame box centres", fontsize=13, fontweight="bold")

    ax[0, 0].imshow(sample.image_left)
    ax[0, 0].set_title("Left RGB", fontsize=10)
    ax[0, 0].axis("off")

    _panel(ax[0, 1], inter["bev_camera"].abs().mean(0).numpy(),
           f"Camera BEV ({inter['bev_camera'].shape[0]} ch, |mean|)", "plasma")
    _panel(ax[0, 2], inter["bev_lidar"].abs().mean(0).numpy(),
           f"LiDAR BEV ({inter['bev_lidar'].shape[0]} ch, |mean|)", "viridis")
    _panel(ax[1, 0], inter["fused"].abs().mean(0).numpy(),
           f"Fused BEV ({inter['fused'].shape[0]} ch, |mean|)", "magma")
    _panel(ax[1, 1], inter["heatmap"].sigmoid().amax(0).numpy(),
           "Head heatmap (max class, σ)", "inferno")
    _panel(ax[1, 2], inter["offset"].norm(dim=0).numpy(),
           "Offset magnitude", "cividis")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return save_path
    plt.show()



# Utils for preprocessing visualizations


def visualize_bev(sample: StereoSample, cmap: str = "inferno") -> None:
    """Show a top-down LiDAR density BEV with the GT box outlines on top.

    The grid (x/y range, resolution) is taken from :class:`PillarConfig` so the
    picture is aligned with the LiDAR/camera BEV branches. Boxes are drawn from
    ``boxes_3d_ego`` — the ego-frame copy that matches the grid; ``boxes_3d`` is
    in the global frame and would land in the wrong cells here.
    """
    cfg = PillarConfig()
    x_min, x_max = cfg.x_range
    y_min, y_max = cfg.y_range
    res = cfg.pillar_size
    nx, ny = cfg.grid_size

    # Point density per cell (ix along x/forward, iy along y/lateral).
    grid = np.zeros((nx, ny), dtype=np.float32)
    pts = sample.lidar_xyz
    if pts.shape[0]:
        m = (pts[:, 0] >= x_min) & (pts[:, 0] < x_max) & (pts[:, 1] >= y_min) & (pts[:, 1] < y_max)
        p = pts[m]
        ix = ((p[:, 0] - x_min) / res).astype(np.int64)
        iy = ((p[:, 1] - y_min) / res).astype(np.int64)
        np.add.at(grid, (ix, iy), 1.0)

    plt.figure(figsize=(6, 8))
    # extent lets us plot boxes directly in metres; origin="lower" -> x up, y right.
    plt.imshow(np.log1p(grid), origin="lower", cmap=cmap,
               extent=[y_min, y_max, x_min, x_max], aspect="auto")
    boxes = sample.boxes_3d_ego
    if boxes is not None and len(boxes) > 0:
        for b in boxes:
            cx, cy, l, w = b[0], b[1], b[7], b[8]
            qw, qx, qy, qz = b[3], b[4], b[5], b[6]
            heading = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            cos_h, sin_h = np.cos(heading), np.sin(heading)
            dx = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2])
            dy = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2])
            xs = cx + dx * cos_h - dy * sin_h
            ys = cy + dx * sin_h + dy * cos_h
            plt.plot(ys, xs, "r-", lw=1)  # (lateral, forward) to match extent
    plt.xlabel("Y lateral (m)")
    plt.ylabel("X forward (m)")
    plt.title("LiDAR density BEV + GT boxes (ego frame)")
    plt.show()


def visualize_encoded_targets(sample: StereoSample, save_path: str | None = None):
    """End-to-end sanity panel for the BEV target encoder + decoder.

    Four panels, all in the ego BEV frame (Y lateral horizontal, X forward
    vertical) so they overlay:

    1. left RGB image (the scene),
    2. LiDAR density BEV with oriented GT boxes (where the objects really are),
    3. the **encoded** heatmap (max over classes) with GT centres,
    4. the **decoded** predictions vs GT centres over the LiDAR BEV.

    If the Gaussian blobs and the decoded ``x`` markers land on the LiDAR
    objects / GT rings, the encode→decode round-trip and the frames all agree.
    Pass ``save_path`` to write a PNG instead of showing interactively.
    """
    import torch

    import globals as G
    from train import TargetEncoder
    from evaluation import CenterPointDecoder

    cfg = PillarConfig()
    x_min, x_max = cfg.x_range
    y_min, y_max = cfg.y_range
    res = cfg.pillar_size
    nx, ny = cfg.grid_size
    extent = [y_min, y_max, x_min, x_max]
    colors = np.array([[0.15, 0.6, 1.0], [1.0, 0.55, 0.1], [0.2, 1.0, 0.3]])

    def _cls_color(arr):
        return colors[np.clip(np.asarray(arr, int), 0, len(colors) - 1)]

    # --- GT boxes: remap labels, keep in-grid + kept classes --------------
    boxes = sample.boxes_3d_ego
    idx = [G.class_index(l) for l in sample.boxes_3d_labels]
    keep = np.array([(c is not None) and (x_min <= b[0] < x_max) and
                     (y_min <= b[1] < y_max) for b, c in zip(boxes, idx)])
    centres = np.array([[b[0], b[1]] for b, k in zip(boxes, keep) if k]).reshape(-1, 2)

    # --- encode + decode --------------------------------------------------
    bx = torch.as_tensor(boxes, dtype=torch.float32)[torch.from_numpy(keep)]
    lab = torch.tensor([c for c, k in zip(idx, keep) if k], dtype=torch.long)
    heatmap, offset = TargetEncoder().encode(bx, lab)
    pred = CenterPointDecoder(score_threshold=0.5)(
        heatmap.unsqueeze(0), offset.unsqueeze(0))[0]
    dec = pred["boxes_2d"].numpy().reshape(-1, 2)      # (M, 2) [x, y]
    dec_cls = pred["classes"].numpy()

    # --- LiDAR density ----------------------------------------------------
    grid = np.zeros((nx, ny), np.float32)
    pts = sample.lidar_xyz
    m = ((pts[:, 0] >= x_min) & (pts[:, 0] < x_max) &
         (pts[:, 1] >= y_min) & (pts[:, 1] < y_max))
    p = pts[m]
    np.add.at(grid, (((p[:, 0] - x_min) / res).astype(int),
                     ((p[:, 1] - y_min) / res).astype(int)), 1.0)

    fig, ax = plt.subplots(2, 2, figsize=(15, 13))

    # 1. RGB
    ax[0, 0].imshow(sample.image_left)
    ax[0, 0].set_title("Left RGB")
    ax[0, 0].axis("off")

    # 2. LiDAR + oriented GT boxes
    ax[0, 1].imshow(np.log1p(grid), origin="lower", cmap="bone", extent=extent,
                    aspect="auto")
    for b, c in zip(boxes, idx):
        if c is None or not (x_min <= b[0] < x_max and y_min <= b[1] < y_max):
            continue
        cx, cy, l, w = b[0], b[1], b[7], b[8]
        qw, qx, qy, qz = b[3], b[4], b[5], b[6]
        h = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        ch, sh = np.cos(h), np.sin(h)
        dx = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2])
        dy = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2])
        xs, ys = cx + dx * ch - dy * sh, cy + dx * sh + dy * ch
        ax[0, 1].plot(ys, xs, "-", color=colors[c], lw=1.3)
    ax[0, 1].set_title("LiDAR density BEV + GT boxes")
    ax[0, 1].set_xlabel("Y lateral (m)")
    ax[0, 1].set_ylabel("X forward (m)")

    # 3. encoded heatmap + oriented GT 3D boxes (to compare blob vs footprint)
    ax[1, 0].imshow(heatmap.amax(0).numpy(), origin="lower", cmap="hot",
                    extent=extent, aspect="auto")
    for b, c in zip(boxes, idx):
        if c is None or not (x_min <= b[0] < x_max and y_min <= b[1] < y_max):
            continue
        cx, cy, l, w = b[0], b[1], b[7], b[8]
        qw, qx, qy, qz = b[3], b[4], b[5], b[6]
        h = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        ch, sh = np.cos(h), np.sin(h)
        dx = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2])
        dy = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2])
        xs, ys = cx + dx * ch - dy * sh, cy + dx * sh + dy * ch
        ax[1, 0].plot(ys, xs, "-", color="cyan", lw=1.0)
    ax[1, 0].set_title("Encoded heatmap (max over classes) + GT 3D boxes")
    ax[1, 0].set_xlabel("Y lateral (m)")
    ax[1, 0].set_ylabel("X forward (m)")

    # 4. decoded vs GT over LiDAR
    ax[1, 1].imshow(np.log1p(grid), origin="lower", cmap="bone", extent=extent,
                    aspect="auto")
    if len(centres):
        ax[1, 1].scatter(centres[:, 1], centres[:, 0], s=95, facecolors="none",
                         edgecolors="cyan", lw=1.5, label="GT centre")
    if len(dec):
        ax[1, 1].scatter(dec[:, 1], dec[:, 0], s=28, c=_cls_color(dec_cls),
                         marker="x", lw=1.5, label="decoded")
    ax[1, 1].legend(loc="upper right")
    ax[1, 1].set_title(f"Decoded {len(dec)} vs {len(centres)} GT centres")
    ax[1, 1].set_xlabel("Y lateral (m)")
    ax[1, 1].set_ylabel("X forward (m)")

    handles = [plt.Line2D([0], [0], marker="s", color="w", markersize=11,
                          markerfacecolor=colors[i], label=name)
               for i, name in enumerate(G.CLASSES)]
    fig.legend(handles=handles, loc="lower center", ncol=len(G.CLASSES),
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=90, bbox_inches="tight")
        plt.close(fig)
        return save_path
    plt.show()

def visualize_frustum(sample: StereoSample, box_index: int = 0) -> None:
    """Show frustum points colored by depth or intensity."""
    fpts = frustum_points(sample, box_index=box_index)
    if fpts.shape[0] == 0: 
        print("No points in frustum")
        return
    pts = fpts[:, :3]
    colors = np.ones_like(pts)
    if fpts.shape[1] >= 6:
        colors = fpts[:, 3:]  # RGB
    else:
        colors = (pts[:, 2:] - pts[:, 2].min()) / (pts[:, 2].max() - pts[:, 2].min() + 1e-6)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=colors, s=5)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("Frustum Points")
    plt.show()

def visualize_voxels(sample: StereoSample) -> None:
    """Show voxel grid as a sparse occupancy plot (requires more code for 3D scatter)."""
    vox = voxel_grid(sample)
    # Simple occupancy visualization (not full 3D scatter for brevity)
    plt.imshow(vox.features[0].max(axis=0), cmap="gray")
    plt.title("Voxel Occupancy (Max Intensity)")
    plt.show()

def visualize_clusters(sample: StereoSample) -> None:
    """Show points colored by cluster labels on a 2D projection."""
    lbls, stats = cluster_points(sample)
    pts = sample.lidar_xyz
    # Project to 2D BEV-like for visualization
    plt.scatter(pts[:, 0], pts[:, 1], c=lbls, s=10)
    plt.title("Clustered Points (2D Projection)")
    plt.show()

def visualize_images(sample: StereoSample) -> None:
    """Show the left and right stereo images side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if sample.image_left is not None:
        axes[0].imshow(sample.image_left)
    axes[0].set_title("Left Image")
    axes[0].axis('off')
    
    if sample.image_right is not None:
        axes[1].imshow(sample.image_right)
    axes[1].set_title("Right Image")
    axes[1].axis('off')
        
    plt.tight_layout()
    plt.show()

def visualize_pointcloud(sample: StereoSample) -> None:
    """Show the full 3D LiDAR point cloud."""
    pts = sample.lidar_xyz
    if pts.shape[0] == 0:
        print("No LiDAR points to show")
        return
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    colors = (pts[:, 2] - pts[:, 2].min()) / (pts[:, 2].max() - pts[:, 2].min() + 1e-6)
    step = max(1, pts.shape[0] // 30000)  # Subsample to keep rendering fast
    
    ax.scatter(pts[::step, 0], pts[::step, 1], pts[::step, 2], c=colors[::step], s=1, cmap='viridis')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"LiDAR Point Cloud (Ego Frame)")
    
    plt.show()


# --------------------------------------------------------------------------- #
# Pipeline C — attention gate visualisation
# --------------------------------------------------------------------------- #
def visualize_attention_gate(
        pipeline_c,
        sample: StereoSample,
        device=None,
        save_path: str | None = None,
):
    """Visualise the learned near/far attention gate of :class:`network.PipelineC`.

    Runs one forward pass through Pipeline C, captures the sigmoid gate
    ``g ∈ (0,1)`` produced by ``fusion.gate``, and renders it on the ego BEV
    grid with::

        g ≈ 0  (dark green)  → the model trusts the stereo camera (near cells)
        g ≈ 1  (dark red)    → the model trusts the LiDAR attention (far cells)

    Overlaid:
    * cyan rings — GT box centres (ego frame, in-grid).
    * dashed white arcs — iso-range circles at 10 / 20 / 30 / 40 m so the
      near/far structure is immediately legible.
    * per-cell gate value on a shared 0–1 colour scale so comparisons across
      runs / training stages are meaningful.

    Parameters
    ----------
    pipeline_c : :class:`network.PipelineC`
        The trained (or untrained) Pipeline C instance.
    sample     : StereoSample
        One frame to run the forward pass on.
    device     : torch.device, optional
        Defaults to CUDA if available.
    save_path  : str, optional
        Write a PNG instead of calling ``plt.show()``.
    """
    import torch
    import globals as G

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipeline_c.to(device).eval()

    # ---- capture gate output via forward hook ----
    gate_map = {}
    def _grab_gate(m, inp, out):
        gate_map["gate"] = out[0, 0].detach().cpu().numpy()  # (nx, ny)
    handle = pipeline_c.detector.fusion.gate.register_forward_hook(_grab_gate)
    with torch.no_grad():
        pipeline_c(sample, device=device)
    handle.remove()

    if "gate" not in gate_map:
        print("Gate not captured — is this really a PipelineC instance?")
        return
    gate = gate_map["gate"]

    # ---- BEV grid geometry ----
    (x_min, x_max), (y_min, y_max) = G.X_RANGE, G.Y_RANGE
    extent = [y_min, y_max, x_min, x_max]
    nx, ny = G.GRID_SIZE
    res = G.BEV_RES_M

    # ---- GT centres (ego frame, in-grid) ----
    centres = (sample.boxes_3d_ego[:, :2]
               if len(sample.boxes_3d_ego) else np.zeros((0, 2)))
    m = ((centres[:, 0] >= x_min) & (centres[:, 0] < x_max) &
         (centres[:, 1] >= y_min) & (centres[:, 1] < y_max))
    centres = centres[m]

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle(
        f"Pipeline C — learned near/far gate | "
        f"{sample.dataset} iter={sample.iteration}",
        fontsize=13, fontweight="bold")

    for ax, cmap, title in zip(
        axes,
        ["RdYlGn_r", "coolwarm"],
        [
            "Gate g  (green=trust stereo, red=trust LiDAR)",
            "Gate g  (blue=stereo, red=LiDAR)",
        ],
    ):
        im = ax.imshow(gate, origin="lower", cmap=cmap, vmin=0.0, vmax=1.0,
                       extent=extent, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="gate g")

        # iso-range arcs from ego origin (x=0, y=0)
        for r in [10, 20, 30, 40]:
            theta = np.linspace(np.radians(-90), np.radians(90), 200)
            arc_x = r * np.cos(theta)
            arc_y = r * np.sin(theta)
            # keep only arcs inside the BEV extent
            inside = ((arc_x >= x_min) & (arc_x < x_max) &
                      (arc_y >= y_min) & (arc_y < y_max))
            if inside.any():
                ax.plot(arc_y[inside], arc_x[inside], "w--", lw=0.8, alpha=0.6)
                # label the arc at the rightmost visible point
                label_idx = np.where(inside)[0][-1]
                ax.text(arc_y[label_idx], arc_x[label_idx],
                        f"{r} m", color="white", fontsize=7, alpha=0.8,
                        ha="center", va="bottom")

        if len(centres):
            ax.scatter(centres[:, 1], centres[:, 0], s=50,
                       facecolors="none", edgecolors="cyan", lw=1.5,
                       label="GT centres")
            ax.legend(loc="upper right", fontsize=8)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Y lateral (m)")
        ax.set_ylabel("X forward (m)")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return save_path
    plt.show()


def compare_pipeline_a_c(
        pipeline_a,
        pipeline_c,
        sample: StereoSample,
        device=None,
        save_path: str | None = None,
):
    """Side-by-side debug forward of Pipeline A and Pipeline C on one frame.

    Four panels: camera BEV | LiDAR BEV | Pipeline A fused BEV (concat+conv) |
    Pipeline C fused BEV (cross-attention). All collapsed to channel-|mean| for
    visualisation. GT centres overlaid in cyan so alignment is clear.

    Both pipelines must already be in eval mode or will be set to eval here.
    """
    import torch
    import globals as G

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline_a.to(device).eval()
    pipeline_c.to(device).eval()

    # Reuse Pipeline A's stereo-depth cache to avoid computing it twice
    pipeline_a.debug = True
    pipeline_c.debug = True

    with torch.no_grad():
        pipeline_a(sample, device=device)
        pipeline_c(sample, device=device)

    inter_a = pipeline_a.intermediates
    inter_c = pipeline_c.intermediates

    (x_min, x_max), (y_min, y_max) = G.X_RANGE, G.Y_RANGE
    extent = [y_min, y_max, x_min, x_max]

    centres = (sample.boxes_3d_ego[:, :2]
               if len(sample.boxes_3d_ego) else np.zeros((0, 2)))
    m = ((centres[:, 0] >= x_min) & (centres[:, 0] < x_max) &
         (centres[:, 1] >= y_min) & (centres[:, 1] < y_max))
    centres = centres[m]

    def _panel(ax, feat, title, cmap):
        lo = float(np.percentile(feat, 1))
        hi = float(np.percentile(feat, 99))
        if hi <= lo:
            lo, hi = float(feat.min()), float(feat.max()) + 1e-9
        im = ax.imshow(feat, origin="lower", cmap=cmap, vmin=lo, vmax=hi,
                       extent=extent, aspect="auto")
        if len(centres):
            ax.scatter(centres[:, 1], centres[:, 0], s=30,
                       facecolors="none", edgecolors="cyan", lw=1.0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Y lateral (m)")
        ax.set_ylabel("X forward (m)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig, ax = plt.subplots(1, 4, figsize=(24, 7))
    fig.suptitle(
        f"Pipeline A vs C — fused BEV comparison | "
        f"{sample.dataset} iter={sample.iteration}",
        fontsize=13, fontweight="bold")

    _panel(ax[0], inter_a["bev_camera"].abs().mean(0).numpy(),
           f"Camera BEV ({inter_a['bev_camera'].shape[0]} ch, |mean|)", "plasma")
    _panel(ax[1], inter_a["bev_lidar"].abs().mean(0).numpy(),
           f"LiDAR BEV ({inter_a['bev_lidar'].shape[0]} ch, |mean|)", "viridis")
    _panel(ax[2], inter_a["fused"].abs().mean(0).numpy(),
           f"Pipeline A fused ({inter_a['fused'].shape[0]} ch, concat+conv)", "magma")
    _panel(ax[3], inter_c["fused"].abs().mean(0).numpy(),
           f"Pipeline C fused ({inter_c['fused'].shape[0]} ch, cross-attn)", "magma")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return save_path
    plt.show()
