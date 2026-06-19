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