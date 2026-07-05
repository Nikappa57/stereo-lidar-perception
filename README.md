# stereo-lidar-perception

**Stereo + LiDAR fusion for 3D object detection in bird's-eye view.**

A Computer Vision course project (AIRO master's program, Sapienza University of Rome) by three members of the [Fast Charge](https://fastcharge.diag.uniroma1.it/) Formula Student Driverless team. The method is developed and benchmarked on **Argoverse 2** and designed to port to the team's driverless car for cone detection.

---

## Idea

Combine a stereo camera and a LiDAR into a single top-down (bird's-eye-view) detector that outputs each object's ground position `(x, y)` and class — exactly what mapping, SLAM and motion planning consume.

Two principles drive the design:

- **Fuse only where data are aligned.** RGB and stereo depth share the image plane, so they fuse in 2D; the LiDAR lives in its own frame, so the two sensors only meet in a common **bird's-eye-view (BEV)** grid.
- **Use measured depth, not predicted depth.** Most camera–LiDAR detectors fuse LiDAR with a *monocular* camera and must estimate per-pixel depth to lift features into 3D — an error-prone step. We have **stereo** depth, so each camera feature is placed into its correct BEV cell by geometry rather than by a learned guess. This is the core novelty.

## Architecture

The main pipeline (mid fusion) has two branches that meet once, in BEV. Each
stage below maps to one block in [`network.py`](network.py):

1. **Camera backbone** — turns the left image into a grid of semantic features.
2. **Splat to BEV** — places those features on the ground plane using stereo depth (no learnable parameters; pure geometry).
3. **LiDAR stem** — encodes the LiDAR BEV map into features (already top-down).
4. **Fusion** — both branches now share the grid, so they are fused before the head. This block is swappable: **Pipeline A/B** uses channel concatenation + 2D convolutions (`ConcatConvFusion`), while **Pipeline C** drops in a **Windowed Cross-Attention Transformer** (`CrossAttentionFusion`). See [`docs/newnetwork.md`](docs/newnetwork.md) for the complete mathematical and architectural breakdown.
5. **BEV backbone** — 2D context reasoning over the fused grid.
6. **Center head** — per-class heatmap + sub-cell offset, decoded into an object list.

## Architectures explored

We compare four fusion strategies along a single axis — *where the two sensors meet*. The lower the purple block, the later the fusion and the more independent the branches.

| A — Mid fusion: image + BEV *(primary)* | B — + painted LiDAR range |
| :---: | :---: |
| <img src="docs/img/pipeline_a.svg" width="380"> | <img src="docs/img/pipeline_b.svg" width="380"> |
| **C — Cross-attention fusion** | **D — Late fusion (baseline)** |
| <img src="docs/img/pipeline_c.svg" width="380"> | <img src="docs/img/pipeline_d.svg" width="380"> |

### The CNN vs. Transformer Fusion Dichotomy
While **Pipeline A/B** relies on $3 \times 3$ convolutions (which assume strict cell-by-cell spatial alignment between camera and LiDAR BEV maps), **Pipeline C** introduces a Transformer-based cross-attention module (`CrossAttentionFusion`) designed to relax alignment constraints and adapt to range-dependent sensor uncertainty:
- **Windowed Cross-Attention ($O(N \cdot \text{win}^2)$):** The BEV grid is partitioned into local $8 \times 8$ windows ($2\text{ m} \times 2\text{ m}$ patches). Camera tokens act as **queries ($Q$)** searching over LiDAR **keys/values ($K, V$)**, reducing attention complexity by **500×** compared to global attention to ensure real-time execution on the Jetson AGX Orin.
- **Swin-Style Relative Position Bias:** A learnable bias table indexed by relative token offsets $(\Delta\text{row}, \Delta\text{col})$ is added to attention dot-products, preserving BEV translation equivariance while guiding attention across stereo depth uncertainty.
- **Learnable Near/Far Spatial Gating ($g \in (0, 1)$):** A per-cell sigmoid gate modulates between camera projection and attended LiDAR features ($F_{\text{fused}} = (1 - g) \cdot F_{\text{cam}} + g \cdot F_{\text{attn}}$). In the near field ($\le 20\text{ m}$), dense stereo depth is trusted ($g \to 0$); in the far field ($> 50\text{ m}$ where stereo disparity error degrades quadratically), the network automatically shifts trust to the attended LiDAR signal ($g \to 1$).

## Dataset

We use **Argoverse 2** through the [`py123d`](https://pypi.org/project/py123d/) loader. Among large autonomous-driving datasets it is the only one that combines everything this project needs:

- a real **stereo** camera pair (most AV datasets are mono-camera),
- a dense **LiDAR** (two aggregated 32-beam sweeps),
- a genuine small-object class — **`CONSTRUCTION_CONE`** — the closest available proxy to Formula Student track cones,
- a **distance-based** detection metric that matches a centre-mapping task and transfers cleanly to cones.

**Classes used:** `REGULAR_VEHICLE` (development and stability), `PEDESTRIAN`, and `CONSTRUCTION_CONE` (transfer-relevant small object).

## Installation

### 1. Install the dataset library

Install `py123d` with the dataset backend (Argoverse 2):

```bash
pip install "py123d[av2]"
```

### 2. Download the dataset

Set the destination directory:

```bash
export AV2_DATA_ROOT=/path/to/argoverse
```

Download a subset of logs (e.g. 5 from the validation split) to test the code:

```bash
py123d-download dataset=av2-sensor \
    'dataset.downloader.splits=[av2-sensor_val]' \
    dataset.downloader.num_logs=5
```

### 3. Convert the data

`py123d` uses Apache Arrow for fast loading. Convert the downloaded data:

```bash
py123d-conversion dataset=av2-sensor \
    'dataset.parser.splits=[av2-sensor_val]'
```

> **`AV2_DATA_ROOT` must still be exported** (step 2) — the conversion CLI reads it
> to locate the raw `sensor/` directory and fails with `av2_sensor_root … None/sensor`
> otherwise.
>
> **Restrict `dataset.parser.splits` to what you actually downloaded.** The parser
> scans `train`/`val`/`test` by default; with only `av2-sensor_val` on disk, the
> override above avoids a `No such file or directory: …/sensor/train` error.

> The `dataset=av2-sensor-stream` option downloads and parses logs on the fly if disk space is limited.

## Usage

Set the workspace path before running:

```bash
export PY123D_DATA_ROOT="/your/data/path"
```

The `py123d` Scene API gives frame-by-frame access:

```python
scene_api.get_lidar_at_iteration(iteration, "lidar_top")        # LiDAR point cloud
scene_api.get_camera_at_iteration(iteration, "pcam_stereo_l")   # stereo camera
scene_api.get_box_detections_se3_at_iteration(iteration)        # 3D box labels
scene_api.get_ego_state_se3_at_iteration(iteration)             # ego pose
```

## Code structure

Six modules (the prescribed layout):

| File | Role |
| --- | --- |
| `globals.py` | Single source of truth: shared BEV grid, channel contract, classes. |
| `utils.py` | Visualization helpers (LiDAR density BEV + GT boxes, frustum, clusters). |
| `data.py` | Dataset loading (`StereoSample`) **and** the geometric preprocessing representations: stereo depth/BEV, voxel grid, frustum points, clustering. |
| `network.py` | Full architecture — one block per diagram node: camera branch (Mono/Stereo BEV), LiDAR stem (PointPillars), fusion, BEV backbone, CenterPoint head. See [`docs/newnetwork.md`](docs/newnetwork.md). |
| `train.py` | BEV target encoder (`TargetEncoder`), CenterPoint loss (Gaussian-focal heatmap + masked L1 offset), LiDAR-only detector + single-frame overfit harness. Multi-frame training loop *(TODO)*. |
| `evaluation.py` | `CenterPointDecoder` (max-pool NMS → metric `(x, y)` + class + score). Distance-AP / CDS metrics *(TODO)*. |

## Evaluation

Two single-sensor baselines (camera-only and LiDAR-only) set the floor — fusion must beat both. Detection is scored with the Argoverse 2 **distance-AP** metric (true positives at 0.5 / 1 / 2 / 4 m) plus the composite detection score, reported per class and stratified by range.

## References

- **BEVFusion** — Multi-Task Multi-Sensor Fusion with Unified BEV Representation. [arXiv:2205.13542](https://arxiv.org/abs/2205.13542)
- **SLBEVFusion** — 3D detection using stereo camera and LiDAR fusion with BEV (Neurocomputing, 2024).
- **FutrTrack** — Camera-LiDAR Fusion Transformer for 3D MOT. [arXiv:2510.19981](https://arxiv.org/abs/2510.19981)
- **Argoverse 2** — Next Generation Datasets for Self-Driving Perception and Forecasting. [arXiv:2301.00493](https://arxiv.org/abs/2301.00493)

## Authors

Leonardo Galgano · Lorenzo Gaudino · Vittorio Cava — Sapienza University of Rome, Fast Charge Driverless.
