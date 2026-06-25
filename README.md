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
- **Bird's-Eye-View (BEV):** Density and height maps projected from the LiDAR point cloud onto a regular 2D grid. (TODO: STEREO BEV)
- **Camera Frustum:** Extracts LiDAR returns that fall inside a 2D camera detection box, lifted into a local frustum frame, optionally with RGB colors appended.
- **Voxel Grid:** Builds a volumetric occupancy and feature grid from the ego-frame point cloud. Includes features like mean height, point density, and intensity.
- **Clustering:** Euclidean-distance DBSCAN clustering of the point cloud, providing per-point labels and cluster statistics (centroids, extents).

### PointPillars (`pointpillars.py`)
The LiDAR branch utilizes a PointPillars architecture to encode raw point clouds into a Bird's-Eye-View (BEV) feature map. The pipeline includes:
1. **Pillarization:** Bins the 3D point cloud into vertical columns (pillars) on an x-y grid. Points are augmented with offsets to their pillar centroid and geometric cell center.
2. **Pillar Feature Net (PFN):** A simplified PointNet (shared MLP and max-pooling) that computes a single feature vector for each pillar.
3. **Scatter:** Scatters the extracted pillar features back onto the dense 2D grid to form a pseudo-image.
4. **BEV Backbone 2D:** A lightweight 2D Convolutional Neural Network that refines the pseudo-image, adding spatial context among neighboring pillars.