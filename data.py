"""Dataset loading for the stereo-lidar-perception project.

This module wraps the unified ``py123d`` Scene API so that the *same* loading
code works for every dataset py123d supports (Argoverse 2, nuPlan, nuScenes,
Waymo, PandaSet, KITTI-360, CARLA, ...). py123d converts each dataset into a
common Arrow-based format and exposes it through :class:`py123d.api.SceneAPI`,
so nothing here is AV2-specific: switching datasets is purely a matter of the
:class:`~py123d.api.SceneFilter` you pass (``datasets`` / ``split_names``) and
the dataset root paths in the environment.

Typical usage::

    from data import Py123dDataset

    dataset = Py123dDataset(split_names=["av2-sensor_val"])
    frame = dataset[0]
    image = frame.camera("pcam_f0").image          # HxWx3 uint8
    boxes = frame.boxes()                            # BoxDetectionsSE3 (3D labels)
    points = frame.lidar().xyz                       # Nx3 ego-frame point cloud

Dataset paths
-------------
py123d resolves sensor/log/map locations from environment variables (see
``py123d-docs/installation.md``):

* ``PY123D_DATA_ROOT`` -> converted logs (``<root>/logs``) and maps (``<root>/maps``).
* ``AV2_DATA_ROOT``, ``NUPLAN_DATA_ROOT``, ... -> the *original* sensor blobs,
  because the converter by default stores only relative paths to images/lidar.

:func:`configure_dataset_paths` wires these up. It respects any variable you
have already exported and, for the self-contained layout shipped with this
repo, falls back to the project ``data/`` directory (which holds both
``logs/`` and the ``sensor/`` blobs).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Union

from py123d.api import SceneAPI, SceneFilter, get_filtered_scenes
from py123d.common.runtime import DatasetPaths, setup_dataset_paths
from py123d.datatypes.detections import BoxDetectionsSE3
from py123d.datatypes.sensors import Camera, CameraID, Lidar, LidarID
from py123d.datatypes.time import Timestamp
from py123d.datatypes.vehicle_state import EgoStateSE3

# Repo-local fallback: ``data/`` contains both converted ``logs/`` and the
# original ``sensor/`` blobs, so it can serve as every root at once.
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "data"

# Per-dataset "original data root" env vars. py123d derives each dataset's
# sensor root from these (e.g. AV2_DATA_ROOT -> ``<root>/sensor``). Listing them
# here keeps path setup dataset-agnostic.
_DATASET_ROOT_ENV_VARS: Sequence[str] = (
    "AV2_DATA_ROOT",
    "NUPLAN_DATA_ROOT",
    "WOD_PERCEPTION_DATA_ROOT",
    "WOD_MOTION_DATA_ROOT",
    "PANDASET_DATA_ROOT",
    "KITTI360_DATA_ROOT",
    "NUSCENES_DATA_ROOT",
)

CameraIdLike = Union[CameraID, str, int]
LidarIdLike = Union[LidarID, str, int]


def configure_dataset_paths(data_root: Optional[Union[str, Path]] = None) -> Path:
    """Point py123d at the dataset roots and return the resolved data root.

    The data root is taken from (in order): the ``data_root`` argument, the
    ``PY123D_DATA_ROOT`` environment variable, or :data:`DEFAULT_DATA_ROOT`.

    Any already-exported dataset root variable (``AV2_DATA_ROOT`` etc.) is left
    untouched; the rest default to ``data_root`` so the self-contained repo
    layout works out of the box while real installs keep their own paths.

    Must be called before any scene/sensor data is read, because the readers
    look up sensor locations through the process-global dataset paths. It is
    invoked automatically by :class:`Py123dDataset`.
    """
    resolved = Path(data_root or os.environ.get("PY123D_DATA_ROOT") or DEFAULT_DATA_ROOT).resolve()

    os.environ["PY123D_DATA_ROOT"] = str(resolved)
    for env_var in _DATASET_ROOT_ENV_VARS:
        # ``setdefault`` keeps user-provided roots; only fills in the gaps.
        os.environ.setdefault(env_var, str(resolved))

    # Initialise the process-global paths from the (now-populated) environment.
    setup_dataset_paths(DatasetPaths.from_env())
    return resolved


@dataclass(frozen=True)
class Frame:
    """A single synchronized time step within a scene.

    Modalities are loaded lazily on access so that consumers only pay for the
    data they actually use (loading all nine AV2 cameras per frame is costly).
    Every accessor is dataset-agnostic and returns ``None`` for modalities a
    given dataset does not provide.
    """

    scene: SceneAPI
    scene_index: int
    iteration: int

    # -- identity / metadata -------------------------------------------------
    @property
    def dataset(self) -> str:
        return self.scene.dataset

    @property
    def split(self) -> str:
        return self.scene.split

    @property
    def location(self) -> str:
        return self.scene.location

    @property
    def log_name(self) -> str:
        return self.scene.log_name

    @property
    def scene_uuid(self) -> str:
        return self.scene.scene_uuid

    @property
    def timestamp(self) -> Timestamp:
        return self.scene.get_timestamp_at_iteration(self.iteration)

    @property
    def available_camera_ids(self) -> List[CameraID]:
        return self.scene.available_camera_ids

    @property
    def available_lidar_ids(self) -> List[LidarID]:
        return self.scene.available_lidar_ids

    # -- modality accessors --------------------------------------------------
    def camera(self, camera_id: CameraIdLike) -> Optional[Camera]:
        """Return the :class:`Camera` (image + intrinsics/extrinsics) for ``camera_id``."""
        return self.scene.get_camera_at_iteration(self.iteration, camera_id)

    def cameras(self, camera_ids: Optional[Sequence[CameraIdLike]] = None) -> Dict[CameraID, Camera]:
        """Return ``{CameraID: Camera}`` for the requested (or all available) cameras."""
        ids = list(camera_ids) if camera_ids is not None else self.available_camera_ids
        out: Dict[CameraID, Camera] = {}
        for cid in ids:
            camera = self.camera(cid)
            if camera is not None:
                out[CameraID.from_arbitrary(cid)] = camera
        return out

    def lidar(self, lidar_id: Optional[LidarIdLike] = None) -> Optional[Lidar]:
        """Return a :class:`Lidar` point cloud.

        Defaults to the merged point cloud when present, otherwise the first
        available lidar, so callers need not know each dataset's lidar layout.
        """
        if lidar_id is None:
            lidar_id = self._default_lidar_id()
            if lidar_id is None:
                return None
        return self.scene.get_lidar_at_iteration(self.iteration, lidar_id)

    def boxes(self) -> Optional[BoxDetectionsSE3]:
        """Return the 3D bounding-box detections (ground-truth labels), if any."""
        return self.scene.get_box_detections_se3_at_iteration(self.iteration)

    def ego_state(self) -> Optional[EgoStateSE3]:
        """Return the ego-vehicle state (pose, dynamics), if available."""
        return self.scene.get_ego_state_se3_at_iteration(self.iteration)

    def _default_lidar_id(self) -> Optional[LidarID]:
        available = self.available_lidar_ids
        if not available:
            return None
        if LidarID.LIDAR_MERGED in available:
            return LidarID.LIDAR_MERGED
        return available[0]

    def __repr__(self) -> str:
        return (
            f"Frame(dataset={self.dataset!r}, log={self.log_name!r}, "
            f"scene={self.scene_index}, iter={self.iteration})"
        )


class Py123dDataset:
    """A flat, frame-indexed view over a filtered set of py123d scenes.

    Scenes are selected with a :class:`~py123d.api.SceneFilter`; every iteration
    (synchronized time step) of every matching scene becomes one indexable
    :class:`Frame`. The map-style ``__len__`` / ``__getitem__`` interface plugs
    directly into a ``torch.utils.data.DataLoader`` without importing torch
    here, keeping the loader framework-agnostic.

    :param data_root: Root holding converted ``logs/`` (and ``maps/``). Defaults
        to ``PY123D_DATA_ROOT`` or the repo ``data/`` dir.
    :param split_names: Split filter, e.g. ``["av2-sensor_val"]``.
    :param datasets: Dataset-name filter, e.g. ``["av2-sensor"]``.
    :param scene_filter: A fully-built :class:`SceneFilter`; if given, the
        individual filter kwargs above are ignored.
    :param filter_kwargs: Any other :class:`SceneFilter` argument
        (``log_names``, ``scene_uuids``, ``max_num_scenes``,
        ``required_scene_modalities``, ``shuffle``, ...).
    """

    def __init__(
        self,
        data_root: Optional[Union[str, Path]] = None,
        *,
        split_names: Optional[Sequence[str]] = None,
        datasets: Optional[Sequence[str]] = None,
        scene_filter: Optional[SceneFilter] = None,
        **filter_kwargs,
    ) -> None:
        self.data_root = configure_dataset_paths(data_root)

        if scene_filter is None:
            scene_filter = SceneFilter(
                datasets=list(datasets) if datasets is not None else None,
                split_names=list(split_names) if split_names is not None else None,
                **filter_kwargs,
            )
        self.scene_filter = scene_filter

        self.scenes: List[SceneAPI] = get_filtered_scenes(scene_filter, data_root=self.data_root)
        # Flat index: one entry per (scene, iteration) so frames are addressable
        # by a single integer. ``number_of_iterations`` counts current + future.
        self._index: List[tuple[int, int]] = [
            (scene_idx, iteration)
            for scene_idx, scene in enumerate(self.scenes)
            for iteration in range(scene.number_of_iterations)
        ]

    @property
    def scene_count(self) -> int:
        return len(self.scenes)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> Frame:
        scene_index, iteration = self._index[index]
        return Frame(scene=self.scenes[scene_index], scene_index=scene_index, iteration=iteration)

    def __iter__(self) -> Iterator[Frame]:
        for index in range(len(self)):
            yield self[index]

    def get_frame(self, scene_index: int, iteration: int) -> Frame:
        """Return the :class:`Frame` for an explicit ``(scene_index, iteration)``."""
        return Frame(scene=self.scenes[scene_index], scene_index=scene_index, iteration=iteration)

    def frames_in_scene(self, scene_index: int) -> Iterator[Frame]:
        """Iterate the frames of a single scene in capture order."""
        scene = self.scenes[scene_index]
        for iteration in range(scene.number_of_iterations):
            yield Frame(scene=scene, scene_index=scene_index, iteration=iteration)

    def __repr__(self) -> str:
        return (
            f"Py123dDataset(data_root={str(self.data_root)!r}, "
            f"scenes={self.scene_count}, frames={len(self)})"
        )
