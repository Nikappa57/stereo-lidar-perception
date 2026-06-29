# stereo-lidar-perception

## Installation

### 1. Install Dataset Library

Install the `py123d` library with the specific dataset support (e.g., Argoverse 2):

```bash
pip install "py123d[av2]"
```

### 2. Download Dataset

Set the destination directory for the dataset:

```bash
export AV2_DATA_ROOT=/path/to/argoverse
```

Download a subset of logs (e.g., 5 logs from the validation set) to test your code:

```bash
py123d-download dataset=av2-sensor \
    'dataset.downloader.splits=[av2-sensor_val]' \
    dataset.downloader.num_logs=5
```

### 3. Convert Data

The `py123d` library uses Apache Arrow for fast data loading. Convert the downloaded data to this unified format:

```bash
py123d-conversion dataset=av2-sensor
```

*Note: The `dataset=av2-sensor-stream` option can be used to download and parse logs on the fly if disk space is limited.*

## Usage

Set the workspace environment variable before running the code:

```bash
export PY123D_DATA_ROOT="/your/data/path"
```

### Scene API

The `py123d` Scene API allows access to frame-by-frame data. Key methods include:

- **Lidar (Depth):** `scene_api.get_lidar_at_iteration(iteration, "lidar_top")`
- **Cameras:** `scene_api.get_camera_at_iteration(iteration, "pcam_f0")`
- **Bounding Boxes (3D Labels):** `scene_api.get_box_detections_se3_at_iteration(iteration)`
- **Vehicle Position:** `scene_api.get_ego_state_se3_at_iteration(iteration)`

## Code Structure

- **Imports:** Package dependencies.
- **Globals:** Globally accessible configuration variables.
- **Utils:** Helper and support functions.
- **Data:** Dataset loading and preprocessing components.
- **Network:** Neural network architecture definition.
- **Train:** Training loop and optimization procedures.
- **Evaluation:** Model testing and validation routines.

## Perception Pipeline

### Preprocessing (`preprocessing.py`)
The preprocessing module is responsible for transforming raw stereo and LiDAR data into structured representations:
- **Bird's-Eye-View (BEV):** Three pixel-aligned BEV branches on the same grid:
  - **LiDAR BEV** — PointPillars (pillarize → PFN → scatter → BEV backbone).
  - **Camera BEV (MonoBEV)** — Lift-Splat-Shoot from a single RGB image with a *predicted* depth distribution.
  - **Stereo BEV (StereoBEV)** — Lift-Splat from the left-rectified image with a *grounded* SGBM depth map; the camera branch for Pipelines A & C.
- **Camera Frustum:** Extracts LiDAR returns that fall inside a 2D camera detection box, lifted into a local frustum frame, optionally with RGB colors appended.
- **Voxel Grid:** Builds a volumetric occupancy and feature grid from the ego-frame point cloud. Includes features like mean height, point density, and intensity.
- **Clustering:** Euclidean-distance DBSCAN clustering of the point cloud, providing per-point labels and cluster statistics (centroids, extents).

### PointPillars (`pointpillars.py`)
The LiDAR branch utilizes a PointPillars architecture to encode raw point clouds into a Bird's-Eye-View (BEV) feature map. The pipeline includes:
1. **Pillarization:** Bins the 3D point cloud into vertical columns (pillars) on an x-y grid. Points are augmented with offsets to their pillar centroid and geometric cell center.
2. **Pillar Feature Net (PFN):** A simplified PointNet (shared MLP and max-pooling) that computes a single feature vector for each pillar.
3. **Scatter:** Scatters the extracted pillar features back onto the dense 2D grid to form a pseudo-image.
4. **BEV Backbone 2D:** A lightweight 2D CNN that refines the pseudo-image, adding spatial context among neighbouring pillars.

### MonoBEV — Camera Branch (`monobev.py`)
The camera branch implements the **Lift-Splat-Shoot** paradigm to produce a BEV feature map from a single calibrated RGB image, pixel-aligned with the LiDAR BEV for direct fusion:
1. **Shared CNN backbone** (EfficientNet-style, stride 8×) extracts dense image features.
2. **Depth head** predicts a softmax distribution over D=41 depth bins per pixel.
3. **Context head** predicts a C=64 semantic feature vector per pixel.
4. **Outer product** — `context ⊗ depth_dist` — weights each depth bin's feature by its predicted probability, building a `H'×W'×D` frustum cloud.
5. **Lift to 3D:** Every frustum point is unprojected to ego frame via `K⁻¹` and the camera-to-ego extrinsic `T_cam2ego`.
6. **Voxel pooling (Splat):** Frustum features are sum-pooled into BEV cells using the sort + cumulative-sum boundary trick — O(M log M), fully vectorised.
7. **BEV Backbone 2D** refines the splatted map.

**Output:** `(B, C, nx, ny)` aligned with the LiDAR BEV and ready for channel-wise fusion.

### StereoBEV — Grounded Stereo-Depth Branch (`stereobev.py` + `preprocessing.py`)
 Structurally identical to MonoBEV but uses **grounded SGBM metric depth** instead of a predicted depth distribution — no outer product, no depth-bin dimension:
1. Compute metric depth via SGBM (`stereo.stereo_depth`) in the rectified-left frame.
2. Resize rectified left image + depth map to the backbone input resolution.
3. Scale the rectified intrinsics (`P1[:3,:3]`) and build the rectified-left → ego extrinsic (`T_left2ego @ R1ᵀ`).
4. **Shared CNN backbone** (same `_EfficientNetBackbone`) + **context head** → `(B, C, H', W')`.
5. **Grounded back-projection** (`_build_grounded_frustum`): each pixel ray is scaled by its SGBM depth directly; pixels with invalid depth (0) are masked out.
6. **Splat** — same sort+cumsum pooling from `monobev.splat`.
7. **BEV Backbone 2D** refinement.

**Output:** `(B, C, nx, ny)` with `C=64`, pixel-aligned with the LiDAR and MonoBEV BEVs.

| | LiDAR BEV | MonoBEV | StereoBEV |
|---|---|---|---|
| Depth source | measured | predicted | measured (stereo) |
| Frustum | — | `H'×W'×D` | `H'×W'` |
| Output ch | 128 | 64 | 64 |

### Stereo — Depth Branch (`stereo.py`)
The stereo branch turns the raw left/right image pair into metric geometry with classic block matching (no learning):
1. **Rectify** — `cv2.stereoRectify` removes the small residual rotation / principal-point offset between `pcam_stereo_l` and `pcam_stereo_r` so epipolar lines are horizontal.
2. **Disparity** — `cv2.StereoSGBM` matches the rectified pair (matched on a downscaled copy for speed, then scaled back to full res).
3. **Depth** — `depth = fx · baseline / disparity` (fx ≈ 1724 rect, baseline ≈ 0.4996 m).
4. **Lift** — `cv2.reprojectImageTo3D` → 3-D points, transformed rectified-left → left → **ego**, giving a coloured point cloud in the same frame as `lidar_xyz`.
5. **Stereo BEV** — the cloud is binned into a geometric BEV (occupancy, density, max/mean height, mean RGB) on the **same grid** as the LiDAR / camera BEVs.

**Output:** `StereoDepth` (disparity + metric depth) and a `(C, nx, ny)` stereo BEV. Validated against the sparse LiDAR depth: **median error ≈ 0.4 m, ~82 % of pixels within 2 m**.

> **See** [`docs/perception_pipeline.md`](docs/perception_pipeline.md) for the full step-by-step explanation with equations and implementation notes.

### Quick start — running all branches

```python
from preprocessing import (
    PointPillarsBranch, MonoBEV, StereoBEVBranch,
    _lidar_bev, _camera_bev, _stereo_bev, _print_bev_stats,
)
from data import Py123dDataset
import torch

dataset = Py123dDataset(split_names=["av2-sensor_val"])
sample  = dataset[0].to_stereo_sample()
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bev_lidar  = _lidar_bev(sample, device)   # (128, 200, 160)  LiDAR BEV
bev_camera = _camera_bev(sample, device)  # ( 64, 200, 160)  MonoBEV camera BEV
bev_stereo = _stereo_bev(sample, device)  # ( 64, 200, 160)  StereoBEV camera BEV

_print_bev_stats("LiDAR BEV",  bev_lidar.cpu().numpy())
_print_bev_stats("Camera BEV", bev_camera.cpu().numpy())
_print_bev_stats("Stereo BEV", bev_stereo.cpu().numpy())
```

All three tensors share the same `(nx=200, ny=160)` spatial grid at 0.25 m/cell, covering `x∈[0,50] m` forward and `y∈[-20,20] m` lateral — ready for channel-wise concatenation in Pipeline A or cross-attention fusion in Pipeline C.

### BEV Fusion — Detection Head (`network.py`)
The two BEV maps are fused (`ConcatConvFusion` — concat + conv) and read by a 2D `CenterPointHead` that outputs a per-class centre heatmap + sub-cell `(x, y)` offset (no yaw/z). The fusion block has a fixed interface (two BEV maps in, one out), so Pipeline C swaps in `CrossAttentionFusion` as a drop-in replacement.

> **See** [`docs/perception_pipeline.md`](docs/perception_pipeline.md) for full step-by-step explanations with equations and implementation notes, and [`docs/bev_fusion.md`](docs/bev_fusion.md) for the BEV fusion input contract (what Stage A must emit, and why).