# data.py Documentation

The `data.py` module defines a custom PyTorch `Dataset` (`Py123dStereoLidarDataset`) that loads and synchronizes sensor data using the `py123d` library.

## Dataset Compatibility
This module is fully parameterized and universal. It works with **any dataset supported by the `py123d` library** (e.g., Argoverse 2, KITTI-360, etc.) that contains a stereo-lidar setup.

The base dataset class `Py123dStereoLidarDataset` accepts the target `dataset_name`, the `split`, and specific sensor names (`left_camera_name`, `right_camera_name`, `lidar_name`). A convenience function `load_av2_dataset` is also provided with Argoverse 2 defaults for quick usage.

## What it Outputs
For every frame (iteration), the dataset returns a `FrameSample` object containing:

| Output | Type | Details |
|---|---|---|
| **Stereo images** | `(H, W, 3) uint8` | Left and right RGB images |
| **LiDAR points** | `(N, 3) float32` | 3D point cloud, transformed to the global coordinate frame |
| **Depth map** | `(H, W) float32` | Sparse 2D depth map projected from LiDAR onto the left camera; `0` = invalid |
| **3D bounding boxes** | `BBox3D` | Center, quaternion, dimensions, 8 corners, label, track token |
| **2D bounding boxes** | `BBox2D` | Projected 3D corners yielding an axis-aligned bounding box on each stereo camera |
| **Ego position** | `(3,) float64` | Vehicle center in the global frame |
| **Extrinsic calibration** | `ExtrinsicCalibration` | Camera→global, ego→global, camera→IMU (all 4×4 matrices) |

## On-the-fly Preprocessing
The module performs several data transformations "on-the-fly" during training, without saving any intermediate files to disk:
- **LiDAR Coordinate Transformation**: AV2 stores point clouds in the ego-vehicle (IMU) frame. The code transforms them to the global frame using `ego.imu_se3` so they perfectly align with camera extrinsics and object bounding boxes.
- **Depth Map Generation**: Calculates a sparse 2D depth map by projecting the 3D LiDAR points onto the camera's image plane using `camera.project_points_global()`.
- **2D Bounding Box Extraction**: Projects ground-truth 3D bounding box corners onto the stereo cameras. It automatically computes the 2D bounding rectangle and filters out objects that are outside the camera's field of view.
- **Fallback on missing data**: If a frame is missing critical sensor data, the dataset automatically tries the next iteration in the same scene.
