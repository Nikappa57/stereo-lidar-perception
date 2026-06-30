# Visualize preprocessing results

import numpy as np
import matplotlib.pyplot as plt

from data import StereoSample, frustum_points, voxel_grid, cluster_points
from network import PillarConfig



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
