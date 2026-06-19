"""
Universal Dataset loading module for any dataset supported by the py123d API 
(e.g., Argoverse 2, KITTI-360, etc.) that contains a stereo-lidar setup.

This module provides a generic PyTorch `Dataset` (`Py123dStereoLidarDataset`) that 
reads raw, synchronized log data on-the-fly without saving intermediate files to disk.
It can be parameterized with specific dataset and sensor names.

Outputs per frame (`FrameSample`):
    - Stereo Images: Left and right RGB camera images (`uint8` arrays).
    - LiDAR Points: 3D point cloud (`float32` array).
    - Depth Map: Sparse 2D depth map aligned with the left camera (`float32` array).
    - Bounding Boxes: Both 3D (global frame) and 2D (projected pixel coordinates).
    - Ego Position: Vehicle center in the global frame.
    - Calibration: Extrinsic 4x4 transformation matrices (camera-to-global, ego-to-global, etc.).

On-the-fly Preprocessing:
    - LiDAR Coordinate Transformation: Transforms points from the ego-vehicle (IMU) 
      coordinate frame into the global coordinate frame so they perfectly align with 
      camera extrinsics and object bounding boxes.
    - Depth Map Generation: Projects the transformed 3D LiDAR points onto the left 
      stereo camera's image plane to generate a sparse 2D depth map matrix.
    - 2D Bounding Box Extraction: Projects ground-truth 3D bounding box corners onto 
      the stereo cameras to dynamically generate 2D bounding boxes, automatically 
      filtering out objects outside the field of view.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset

from py123d.api import SceneAPI
from py123d.api.scene.arrow.arrow_scene_builder import ArrowSceneBuilder
from py123d.api.scene.scene_filter import SceneFilter
from py123d.common.execution.sequential_executor import SequentialExecutor
from py123d.datatypes import (
    BoxDetectionSE3,
    BoxDetectionsSE3,
    Camera,
    CameraID,
    EgoStateSE3,
    Lidar,
)
from py123d.geometry import PoseSE3


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BBox2D:
    """Axis-aligned 2D bounding box in pixel coordinates."""

    u_min: float
    v_min: float
    u_max: float
    v_max: float
    label: str
    track_token: str

    @property
    def width(self) -> float:
        return self.u_max - self.u_min

    @property
    def height(self) -> float:
        return self.v_max - self.v_min

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_array(self) -> npt.NDArray[np.float64]:
        """Return [u_min, v_min, u_max, v_max]."""
        return np.array([self.u_min, self.v_min, self.u_max, self.v_max], dtype=np.float64)


@dataclass
class BBox3D:
    """Oriented 3D bounding box in global frame."""

    center_xyz: npt.NDArray[np.float64]        # (3,) — x, y, z
    orientation_quat: npt.NDArray[np.float64]   # (4,) — qw, qx, qy, qz
    dimensions: npt.NDArray[np.float64]         # (3,) — length, width, height
    label: str
    track_token: str
    corners_3d: npt.NDArray[np.float64]         # (8, 3) — world-frame corners

    def to_array(self) -> npt.NDArray[np.float64]:
        """Return [x, y, z, qw, qx, qy, qz, l, w, h]."""
        return np.concatenate([self.center_xyz, self.orientation_quat, self.dimensions])


@dataclass
class ObjectDetection:
    """Combined 2D + 3D detection for a single object."""

    bbox_3d: BBox3D
    bbox_2d_left: Optional[BBox2D] = None   # Projected onto left stereo camera
    bbox_2d_right: Optional[BBox2D] = None  # Projected onto right stereo camera


@dataclass
class ExtrinsicCalibration:
    """Extrinsic calibration data for one frame.

    All poses are expressed as 4×4 homogeneous transformation matrices
    mapping *from the sensor/body frame to the global frame*.
    """

    stereo_left_to_global: npt.NDArray[np.float64]   # (4, 4)
    stereo_right_to_global: npt.NDArray[np.float64]  # (4, 4)
    ego_to_global: npt.NDArray[np.float64]            # (4, 4)

    # Static sensor-to-IMU transforms (same for all frames within a log)
    stereo_left_to_imu: npt.NDArray[np.float64]      # (4, 4)
    stereo_right_to_imu: npt.NDArray[np.float64]     # (4, 4)


@dataclass
class FrameSample:
    """All data for a single synchronised frame.

    Fields
    ------
    stereo_left_image : (H, W, 3) uint8 RGB image from the left stereo camera.
    stereo_right_image : (H, W, 3) uint8 RGB image from the right stereo camera.
    lidar_points : (N, 3) float32 point cloud in global frame (x, y, z).
    lidar_depth_map : (H, W) float32 sparse depth map projected onto left stereo
        camera. Invalid pixels are 0.0.
    detections : list of combined 2D/3D detections.
    ego_position : (3,) float64 ego-vehicle center in global frame.
    calibration : extrinsic calibration for this frame.
    log_name : identifier of the driving log.
    iteration : frame index within the scene.
    """

    stereo_left_image: npt.NDArray[np.uint8]                # (H, W, 3)
    stereo_right_image: npt.NDArray[np.uint8]               # (H, W, 3)
    lidar_points: npt.NDArray[np.float32]                   # (N, 3)
    lidar_depth_map: npt.NDArray[np.float32]                # (H, W)
    detections: List[ObjectDetection]
    ego_position: npt.NDArray[np.float64]                   # (3,)
    calibration: ExtrinsicCalibration
    log_name: str
    iteration: int


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def _project_3d_box_to_2d(
    camera: Camera,
    detection: BoxDetectionSE3,
    img_h: int,
    img_w: int,
) -> Optional[BBox2D]:
    """Project an SE3 bounding-box onto a camera and return an axis-aligned 2D bbox.

    Returns None when fewer than 2 corners are visible (cannot form a box).
    The returned box is clamped to image boundaries.
    """
    corners_3d = detection.bounding_box_se3.corners_array  # (8, 3) global frame
    pixels, in_fov, depths = camera.project_points_global(corners_3d)

    # Keep only corners in front of the camera and inside the image FOV
    valid = in_fov & (depths > 0)
    if valid.sum() < 2:
        return None

    valid_px = pixels[valid]
    u_min = float(np.clip(valid_px[:, 0].min(), 0, img_w - 1))
    v_min = float(np.clip(valid_px[:, 1].min(), 0, img_h - 1))
    u_max = float(np.clip(valid_px[:, 0].max(), 0, img_w - 1))
    v_max = float(np.clip(valid_px[:, 1].max(), 0, img_h - 1))

    # Discard degenerate boxes
    if (u_max - u_min) < 1.0 or (v_max - v_min) < 1.0:
        return None

    return BBox2D(
        u_min=u_min,
        v_min=v_min,
        u_max=u_max,
        v_max=v_max,
        label=str(detection.attributes.default_label.name),
        track_token=detection.attributes.track_token,
    )


def _lidar_points_to_global(
    lidar: Lidar,
    ego_state: EgoStateSE3,
) -> npt.NDArray[np.float64]:
    """Transform LiDAR points from ego/IMU frame to global frame.

    AV2 stores LiDAR point clouds in the ego-vehicle (IMU) coordinate frame.
    This function applies the ego IMU-to-global transformation so that the
    points are in the same coordinate system as camera extrinsics and
    bounding-box centres.
    """
    pts_ego = np.asarray(lidar.point_cloud_3d, dtype=np.float64)  # (N, 3)
    ego_T = ego_state.imu_se3.transformation_matrix  # (4, 4) IMU → global
    pts_homo = np.hstack([pts_ego, np.ones((len(pts_ego), 1), dtype=np.float64)])
    pts_global = (ego_T @ pts_homo.T).T[:, :3]
    return pts_global


def _compute_lidar_depth_map(
    camera: Camera,
    points_global: npt.NDArray[np.float64],
    img_h: int,
    img_w: int,
) -> npt.NDArray[np.float32]:
    """Project LiDAR points (in global frame) onto the camera to build a sparse depth map.

    Returns an (H, W) float32 array where non-zero pixels contain the depth
    (distance along the camera's optical axis) of the closest LiDAR point
    that projects to that pixel.
    """
    pixels, in_fov, depths = camera.project_points_global(points_global)

    valid = in_fov & (depths > 0)
    depth_map = np.zeros((img_h, img_w), dtype=np.float32)

    if not valid.any():
        return depth_map

    valid_px = pixels[valid]
    valid_depths = depths[valid].astype(np.float32)

    # Round to integer pixel coordinates
    u = np.clip(np.round(valid_px[:, 0]).astype(np.int32), 0, img_w - 1)
    v = np.clip(np.round(valid_px[:, 1]).astype(np.int32), 0, img_h - 1)

    # When multiple points map to the same pixel, keep the closest one
    for i in range(len(u)):
        current = depth_map[v[i], u[i]]
        if current == 0.0 or valid_depths[i] < current:
            depth_map[v[i], u[i]] = valid_depths[i]

    return depth_map


# ---------------------------------------------------------------------------
# Per-scene frame extraction
# ---------------------------------------------------------------------------

def _extract_frame(
    scene: SceneAPI,
    iteration: int,
    left_camera_name: str,
    right_camera_name: str,
    lidar_name: str,
) -> Optional[FrameSample]:
    """Extract all modalities for a single iteration within a scene.

    Returns None if any critical modality (stereo cameras, LiDAR, ego state)
    is missing for this iteration.
    """
    # --- Stereo cameras ---
    cam_left: Optional[Camera] = scene.get_camera_at_iteration(iteration, left_camera_name)
    cam_right: Optional[Camera] = scene.get_camera_at_iteration(iteration, right_camera_name)
    if cam_left is None or cam_right is None:
        return None

    img_left = cam_left.image   # (H, W, 3) uint8
    img_right = cam_right.image
    img_h, img_w = img_left.shape[:2]

    # --- LiDAR ---
    lidar: Optional[Lidar] = scene.get_lidar_at_iteration(iteration, lidar_name)
    if lidar is None:
        return None
    # --- Ego state (needed before LiDAR transform) ---
    ego_state: Optional[EgoStateSE3] = scene.get_ego_state_se3_at_iteration(iteration)
    if ego_state is None:
        return None
    ego_position = np.array(
        [ego_state.center_se3.x, ego_state.center_se3.y, ego_state.center_se3.z],
        dtype=np.float64,
    )

    # --- LiDAR: transform from ego/IMU frame → global frame ---
    lidar_points_global = _lidar_points_to_global(lidar, ego_state)
    lidar_points = lidar_points_global.astype(np.float32)  # (N, 3) global frame

    # --- Depth map (LiDAR → left stereo camera) ---
    depth_map = _compute_lidar_depth_map(cam_left, lidar_points_global, img_h, img_w)

    # --- Bounding boxes ---
    box_detections: Optional[BoxDetectionsSE3] = scene.get_box_detections_se3_at_iteration(iteration)
    detections: List[ObjectDetection] = []

    if box_detections is not None:
        for det in box_detections:
            bbox_se3 = det.bounding_box_se3
            center = bbox_se3.center_se3

            bbox_3d = BBox3D(
                center_xyz=np.array([center.x, center.y, center.z], dtype=np.float64),
                orientation_quat=np.array([center.qw, center.qx, center.qy, center.qz], dtype=np.float64),
                dimensions=np.array([bbox_se3.length, bbox_se3.width, bbox_se3.height], dtype=np.float64),
                label=str(det.attributes.default_label.name),
                track_token=det.attributes.track_token,
                corners_3d=bbox_se3.corners_array.copy(),
            )

            bbox_2d_left = _project_3d_box_to_2d(cam_left, det, img_h, img_w)
            bbox_2d_right = _project_3d_box_to_2d(cam_right, det, img_h, img_w)

            detections.append(ObjectDetection(
                bbox_3d=bbox_3d,
                bbox_2d_left=bbox_2d_left,
                bbox_2d_right=bbox_2d_right,
            ))

    # --- Extrinsic calibration ---
    calibration = ExtrinsicCalibration(
        stereo_left_to_global=cam_left.camera_to_global_se3.transformation_matrix,
        stereo_right_to_global=cam_right.camera_to_global_se3.transformation_matrix,
        ego_to_global=ego_state.center_se3.transformation_matrix,
        stereo_left_to_imu=cam_left.metadata.camera_to_imu_se3.transformation_matrix,
        stereo_right_to_imu=cam_right.metadata.camera_to_imu_se3.transformation_matrix,
    )

    return FrameSample(
        stereo_left_image=img_left,
        stereo_right_image=img_right,
        lidar_points=lidar_points,
        lidar_depth_map=depth_map,
        detections=detections,
        ego_position=ego_position,
        calibration=calibration,
        log_name=scene.log_name,
        iteration=iteration,
    )


# ---------------------------------------------------------------------------
# Scene index: maps flat dataset indices to (scene, iteration) pairs
# ---------------------------------------------------------------------------

@dataclass
class _SceneEntry:
    """Bookkeeping for one scene inside the dataset."""

    scene: SceneAPI
    start_idx: int                # global index of iteration 0
    num_iterations: int


# ---------------------------------------------------------------------------
# PyTorch dataset
# ---------------------------------------------------------------------------

class Py123dStereoLidarDataset(Dataset):
    """PyTorch-compatible universal dataset over py123d Sensor data.

    Each item is a :class:`FrameSample` containing synchronised stereo images,
    LiDAR, depth, 3D/2D bounding boxes, ego position, and calibration.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g., ``"av2"``, ``"kitti360"``). Used to set the
        correct environment variable (e.g., ``AV2_DATA_ROOT``).
    data_root : str or Path
        Root directory that contains the ``logs/`` (converted Arrow data) and
        ``sensor/`` (raw images/LiDAR files) subdirectories.
    split : str
        Split name, e.g. ``"av2-sensor_val"``, ``"kitti360_train"``.
    left_camera_name : str
        Sensor name for the left camera (e.g., ``"pcam_stereo_l"``).
    right_camera_name : str
        Sensor name for the right camera (e.g., ``"pcam_stereo_r"``).
    lidar_name : str
        Sensor name for the LiDAR (e.g., ``"lidar_top"``).
    log_names : list[str], optional
        If given, restrict to these log UUIDs.
    max_scenes : int, optional
        Cap the number of scenes (logs) loaded.  Useful for debugging.
    """

    def __init__(
        self,
        dataset_name: str,
        data_root: str | Path,
        split: str,
        left_camera_name: str,
        right_camera_name: str,
        lidar_name: str,
        log_names: Optional[List[str]] = None,
        max_scenes: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.dataset_name = dataset_name
        self.left_camera_name = left_camera_name
        self.right_camera_name = right_camera_name
        self.lidar_name = lidar_name

        data_root = Path(data_root).resolve()

        # Ensure the environment variables are set so py123d can find data.
        os.environ.setdefault("PY123D_DATA_ROOT", str(data_root))
        env_var_name = f"{dataset_name.upper()}_DATA_ROOT"
        os.environ.setdefault(env_var_name, str(data_root))

        # Build scenes from the converted Arrow logs.
        builder = ArrowSceneBuilder(
            logs_root=data_root / "logs",
            maps_root=data_root / "maps",
        )
        scene_filter = SceneFilter(
            split_names=[split],
            log_names=log_names,
        )
        executor = SequentialExecutor()
        scenes: List[SceneAPI] = builder.get_scenes(scene_filter, executor)

        if max_scenes is not None:
            scenes = scenes[:max_scenes]

        if len(scenes) == 0:
            raise RuntimeError(
                f"No scenes found for split='{split}' under {data_root}. "
                f"Have you run `py123d-conversion dataset={dataset_name}`?"
            )

        # Build a flat index across all scenes.
        self._entries: List[_SceneEntry] = []
        total = 0
        for sc in scenes:
            n = sc.number_of_iterations
            self._entries.append(_SceneEntry(scene=sc, start_idx=total, num_iterations=n))
            total += n
        self._total_frames = total

        self.split = split
        self.data_root = data_root

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._total_frames

    def _resolve_index(self, idx: int) -> Tuple[SceneAPI, int]:
        """Map a flat index to (scene, iteration)."""
        if idx < 0:
            idx += self._total_frames
        if idx < 0 or idx >= self._total_frames:
            raise IndexError(f"Index {idx} out of range for dataset of size {self._total_frames}")

        # Binary-style search — entries are sorted by start_idx.
        for entry in self._entries:
            if idx < entry.start_idx + entry.num_iterations:
                return entry.scene, idx - entry.start_idx
        raise IndexError(f"Index {idx} could not be resolved")  # Should never happen

    def __getitem__(self, idx: int) -> FrameSample:
        scene, iteration = self._resolve_index(idx)
        sample = _extract_frame(
            scene, 
            iteration, 
            self.left_camera_name, 
            self.right_camera_name, 
            self.lidar_name
        )

        if sample is None:
            # Some frames may lack data (e.g. first/last frames of a log).
            # Try the next valid frame within this scene.
            entry = next(e for e in self._entries if e.scene is scene)
            for fallback in range(iteration + 1, entry.num_iterations):
                sample = _extract_frame(
                    scene, 
                    fallback, 
                    self.left_camera_name, 
                    self.right_camera_name, 
                    self.lidar_name
                )
                if sample is not None:
                    return sample
            raise RuntimeError(
                f"Could not load any valid frame starting from iteration {iteration} "
                f"in log {scene.log_name}"
            )

        return sample

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def num_scenes(self) -> int:
        """Number of driving logs (scenes) in the dataset."""
        return len(self._entries)

    @property
    def scene_log_names(self) -> List[str]:
        """Log UUIDs for every scene."""
        return [e.scene.log_name for e in self._entries]

    def get_scene(self, scene_idx: int) -> SceneAPI:
        """Return the raw :class:`SceneAPI` for a given scene index."""
        return self._entries[scene_idx].scene


# ---------------------------------------------------------------------------
# Convenience loaders
# ---------------------------------------------------------------------------

def load_py123d_dataset(
    dataset_name: str,
    data_root: str | Path,
    split: str,
    left_camera_name: str,
    right_camera_name: str,
    lidar_name: str,
    log_names: Optional[List[str]] = None,
    max_scenes: Optional[int] = None,
) -> Py123dStereoLidarDataset:
    """Create a :class:`Py123dStereoLidarDataset` for any py123d dataset."""
    return Py123dStereoLidarDataset(
        dataset_name=dataset_name,
        data_root=data_root,
        split=split,
        left_camera_name=left_camera_name,
        right_camera_name=right_camera_name,
        lidar_name=lidar_name,
        log_names=log_names,
        max_scenes=max_scenes,
    )


def load_av2_dataset(
    data_root: str | Path = "data",
    split: str = "av2-sensor_val",
    log_names: Optional[List[str]] = None,
    max_scenes: Optional[int] = None,
) -> Py123dStereoLidarDataset:
    """Convenience function to load Argoverse 2 data with correct AV2 defaults."""
    return Py123dStereoLidarDataset(
        dataset_name="av2",
        data_root=data_root,
        split=split,
        left_camera_name="pcam_stereo_l",
        right_camera_name="pcam_stereo_r",
        lidar_name="lidar_top",
        log_names=log_names,
        max_scenes=max_scenes,
    )
