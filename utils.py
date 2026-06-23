# Visualize preprocessing results

from typing import Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from data import StereoSample
from preprocessing import BEVConfig, bev_map, frustum_points, voxel_grid, cluster_points



# Utils for preprocessing visualizations


def visualize_bev(sample: StereoSample, cmap: str = "inferno") -> None:
    """Show BEV map with optional object outlines and points."""
    cfg = BEVConfig()
    bev = bev_map(sample, config=cfg)
    # Normalize for visualization
    bev_vis = bev.copy()
    for c in range(bev_vis.shape[0]):
        bev_vis[c] = (bev_vis[c] - bev_vis[c].min()) / (bev_vis[c].max() - bev_vis[c].min() + 1e-6)
    grid_img = make_grid(torch.from_numpy(bev_vis), nrow=1, normalize=False).permute(1, 2, 0).numpy()
    plt.imshow(grid_img)
    # Overlay boxes if present
    if sample.boxes_3d is not None and len(sample.boxes_3d) > 0:
        for b in sample.boxes_3d:
            l, w = b[7], b[8]
            qw, qx, qy, qz = b[3], b[4], b[5], b[6]
            heading = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            # Draw box edges properly transformed to pixel coords
            cx, cy = b[0], b[1]
            cos_h, sin_h = np.cos(heading), np.sin(heading)
            dx = np.array([l/2, l/2, -l/2, -l/2, l/2])
            dy = np.array([w/2, -w/2, -w/2, w/2, w/2])
            x_corners = cx + dx * cos_h - dy * sin_h
            y_corners = cy + dx * sin_h + dy * cos_h
            px = (x_corners - cfg.x_range[0]) / cfg.resolution
            py = (y_corners - cfg.y_range[0]) / cfg.resolution
            plt.plot(px, py, 'r-', lw=1)
    plt.title("BEV Map")
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
