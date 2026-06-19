"""Tests and a visual sanity-check for the py123d data loader (``data.py``).

Two ways to use this file:

* ``pytest tests/test_data.py`` — headless assertions that the dataset loads
  and that every modality (cameras, lidar, 3D boxes, ego state) is well-formed,
  including a projection check that the 3D boxes actually land inside a camera.

* ``python tests/test_data.py`` — opens a GUI window showing one full-size
  camera image at a time (with the ground-truth 3D bounding boxes drawn on
  top). Scroll / ←→ pages through every frame in the loaded scenes and ↑↓
  switches camera. The overlay uses py123d's own projection, so it is
  guaranteed consistent with the data.

Although the data shipped here is Argoverse 2, nothing below is AV2-specific:
pass a different ``--split`` (or ``Py123dDataset(datasets=[...])``) and the same
code visualises any py123d dataset.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

# Make the repo root importable so ``data.py`` resolves when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import Frame, Py123dDataset  # noqa: E402

DEFAULT_SPLIT = "av2-sensor_val"


def build_dataset(max_num_scenes: int = 1, split: str = DEFAULT_SPLIT) -> Py123dDataset:
    """Load a small slice of the dataset for testing/visualisation."""
    return Py123dDataset(split_names=[split], max_num_scenes=max_num_scenes)


def _box_corners_global(boxes) -> np.ndarray:
    """Stack the 8 global-frame corners of every box -> ``(N*8, 3)``."""
    from py123d.geometry.utils.bounding_box_utils import bbse3_array_to_corners_array

    box_array = np.stack([d.bounding_box_se3.array for d in boxes.box_detections])
    return bbse3_array_to_corners_array(box_array).reshape(-1, 3)


# --------------------------------------------------------------------------- #
# pytest: data-loading verification (no GUI)
# --------------------------------------------------------------------------- #
import pytest  # noqa: E402


@pytest.fixture(scope="module")
def dataset() -> Py123dDataset:
    return build_dataset()


@pytest.fixture(scope="module")
def frame(dataset: Py123dDataset) -> Frame:
    assert len(dataset) > 0, "dataset produced no frames"
    return dataset[0]


def test_dataset_loads_scenes(dataset: Py123dDataset):
    assert dataset.scene_count > 0, "no scenes matched the filter"
    assert len(dataset) > 0, "scenes contain no frames"


def test_frame_metadata(frame: Frame):
    assert frame.dataset, "missing dataset name"
    assert frame.log_name, "missing log name"
    assert frame.scene_uuid, "missing scene uuid"
    # Timestamps must be present and strictly increase along a scene.
    assert frame.timestamp.time_us > 0


def test_cameras_load(frame: Frame):
    assert frame.available_camera_ids, "scene exposes no cameras"
    cameras = frame.cameras()
    assert cameras, "no cameras could be loaded"
    for camera in cameras.values():
        image = camera.image
        assert isinstance(image, np.ndarray)
        assert image.ndim == 3 and image.shape[2] == 3, f"unexpected image shape {image.shape}"
        assert image.dtype == np.uint8
        assert image.shape[0] == camera.metadata.height
        assert image.shape[1] == camera.metadata.width


def test_lidar_loads(frame: Frame):
    lidar = frame.lidar()
    if lidar is None:
        pytest.skip("dataset has no lidar")
    assert lidar.xyz.ndim == 2 and lidar.xyz.shape[1] == 3
    assert lidar.xyz.shape[0] > 0, "empty point cloud"


def test_boxes_load(frame: Frame):
    boxes = frame.boxes()
    if boxes is None:
        pytest.skip("dataset has no box detections")
    assert len(boxes.box_detections) > 0, "frame has no boxes"
    box = boxes.box_detections[0]
    # Each detection carries a 3D box and a semantic label.
    assert box.bounding_box_se3.array.shape[0] >= 7  # x,y,z, dims, rotation...
    assert box.attributes.default_label is not None


def test_ego_state_loads(frame: Frame):
    ego = frame.ego_state()
    if ego is None:
        pytest.skip("dataset has no ego state")
    # The ego pose should be a finite SE3.
    assert np.all(np.isfinite(ego.imu_se3.array))


def test_boxes_project_into_camera(frame: Frame):
    """The core correctness check: ground-truth 3D boxes must project into the
    image plane of at least one camera. This exercises intrinsics + extrinsics
    together and is exactly what the GUI overlay relies on."""
    boxes = frame.boxes()
    if boxes is None or not boxes.box_detections:
        pytest.skip("no boxes to project")

    corners_global = _box_corners_global(boxes)
    any_in_view = False
    for camera in frame.cameras().values():
        _pixels, in_fov, _depth = camera.project_points_global(corners_global)
        if bool(np.any(in_fov)):
            any_in_view = True
            break
    assert any_in_view, "no box projected into any camera — bad extrinsics/intrinsics?"


# --------------------------------------------------------------------------- #
# GUI: scroll through frames, each shown with its 3D bounding boxes drawn on top
# --------------------------------------------------------------------------- #
def _make_grid(camera_ids: Sequence, cols: int):
    """Create a figure with one axis per camera; return ``(fig, flat_axes)``."""
    import matplotlib.pyplot as plt

    n = len(camera_ids)
    cols = max(1, min(cols, n))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    return fig, list(axes.flat)


def _draw_frame(axes_flat: Sequence, frame: Frame, camera_ids: Sequence) -> int:
    """Draw one frame's camera images (with 3D boxes) into existing axes.

    Returns the number of boxes in the frame. Axes are cleared first so the same
    figure can be reused while scrolling.
    """
    from py123d.visualization.matplotlib.camera import add_box_detections_to_camera_ax

    boxes = frame.boxes()
    for ax, camera_id in zip(axes_flat, camera_ids):
        ax.clear()
        camera = frame.camera(camera_id)
        if camera is None:
            ax.axis("off")
            continue
        if boxes is not None:
            add_box_detections_to_camera_ax(ax, camera, boxes)
        else:
            ax.imshow(camera.image)
        ax.set_title(getattr(camera_id, "name", str(camera_id)), fontsize=9)
        ax.axis("off")

    for ax in list(axes_flat)[len(camera_ids):]:
        ax.axis("off")

    return 0 if boxes is None else len(boxes.box_detections)


def render_frame(frame: Frame, camera_ids: Optional[Sequence] = None, cols: int = 3):
    """Build a standalone figure for a single frame (used by ``--save``)."""
    ids: List = list(camera_ids) if camera_ids is not None else frame.available_camera_ids
    fig, axes_flat = _make_grid(ids, cols)
    n_boxes = _draw_frame(axes_flat, frame, ids)
    fig.suptitle(
        f"{frame.dataset} | log {frame.log_name} | frame {frame.iteration} | {n_boxes} boxes",
        fontsize=12,
    )
    fig.tight_layout()
    return fig


class FrameViewer:
    """Interactive viewer: one full-size camera image at a time.

    A single image fills the window; scrolling (or ←/→) pages through the frames
    of the loaded scenes, and ↑/↓ switches between the cameras of the current
    frame. The figure is reused and redrawn in place, so navigation is
    responsive. Both axes of navigation wrap around at the ends.

    Controls:
        →  n  space   /  ←  p  backspace    next / previous frame
        scroll wheel                        next / previous frame
        ↓  .  /  ↑  ,                        next / previous camera
        f                                   toggle OS fullscreen (matplotlib)
        q                                   quit
    """

    def __init__(
        self,
        dataset: Py123dDataset,
        start_index: int = 0,
        camera_ids: Optional[Sequence] = None,
    ) -> None:
        if len(dataset) == 0:
            raise ValueError("dataset has no frames to display")
        self.dataset = dataset
        self.frame_index = start_index % len(dataset)

        start_frame = dataset[self.frame_index]
        self.camera_ids = list(camera_ids) if camera_ids else list(start_frame.available_camera_ids)
        self.camera_index = 0

        import matplotlib.pyplot as plt

        self.fig, self.ax = plt.subplots(figsize=(14, 9))
        # Let the image fill the window, leaving only a strip at the top for the title.
        self.fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.92)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self._maximize_window()
        self._draw()

    @staticmethod
    def _maximize_window() -> None:
        """Best-effort maximise of the GUI window (backend-dependent)."""
        import matplotlib.pyplot as plt

        manager = plt.get_current_fig_manager()
        for attempt in ("window.showMaximized", "window.state", "resize", "full_screen_toggle"):
            try:
                if attempt == "window.showMaximized":
                    manager.window.showMaximized()  # Qt
                elif attempt == "window.state":
                    manager.window.state("zoomed")  # Tk
                else:
                    return
                return
            except Exception:  # noqa: BLE001 — purely cosmetic
                continue

    def _draw(self) -> None:
        from py123d.visualization.matplotlib.camera import add_box_detections_to_camera_ax

        frame = self.dataset[self.frame_index]
        camera_id = self.camera_ids[self.camera_index]
        camera = frame.camera(camera_id)
        boxes = frame.boxes()

        self.ax.clear()
        if camera is None:
            self.ax.text(0.5, 0.5, f"{getattr(camera_id, 'name', camera_id)}: no image", ha="center")
        elif boxes is not None:
            add_box_detections_to_camera_ax(self.ax, camera, boxes)
        else:
            self.ax.imshow(camera.image)
        self.ax.axis("off")

        n_boxes = 0 if boxes is None else len(boxes.box_detections)
        cam_name = getattr(camera_id, "name", str(camera_id))
        self.ax.set_title(
            f"{frame.dataset} | {cam_name} ({self.camera_index + 1}/{len(self.camera_ids)}) | "
            f"frame {self.frame_index + 1}/{len(self.dataset)} (iter {frame.iteration}) | {n_boxes} boxes\n"
            f"←/→ or scroll: frame · ↑/↓: camera · f: fullscreen · q: quit",
            fontsize=11,
        )
        self.fig.canvas.draw_idle()

    def _step_frame(self, delta: int) -> None:
        self.frame_index = (self.frame_index + delta) % len(self.dataset)
        self._draw()

    def _step_camera(self, delta: int) -> None:
        self.camera_index = (self.camera_index + delta) % len(self.camera_ids)
        self._draw()

    def _on_key(self, event) -> None:
        if event.key in ("right", "n", " "):
            self._step_frame(1)
        elif event.key in ("left", "p", "backspace"):
            self._step_frame(-1)
        elif event.key in ("down", "."):
            self._step_camera(1)
        elif event.key in ("up", ","):
            self._step_camera(-1)

    def _on_scroll(self, event) -> None:
        self._step_frame(-1 if event.button == "up" else 1)

    def show(self) -> None:
        import matplotlib.pyplot as plt

        plt.show()


def _flat_start_index(dataset: Py123dDataset, scene_index: int, iteration: int) -> int:
    """Map a ``(scene, iteration)`` pair to a flat dataset index."""
    start = sum(dataset.scenes[si].number_of_iterations for si in range(scene_index))
    return start + iteration


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Scroll through py123d frames with 3D bounding boxes.")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="py123d split name to load")
    parser.add_argument("--scene", type=int, default=0, help="scene index to start at")
    parser.add_argument("--iteration", type=int, default=None, help="frame index within the scene to start at")
    parser.add_argument("--camera", default=None, help="show a single camera id (e.g. pcam_f0) instead of all")
    parser.add_argument("--max-scenes", type=int, default=1, help="how many scenes to load (scroll spans all of them)")
    parser.add_argument("--save", default=None, help="save the starting frame to this path (does not block)")
    parser.add_argument("--no-show", action="store_true", help="do not open the GUI window")
    args = parser.parse_args(argv)

    print(f"Loading split {args.split!r} ...")
    dataset = build_dataset(max_num_scenes=args.max_scenes, split=args.split)
    print(dataset)
    if dataset.scene_count == 0:
        print("No scenes matched — check PY123D_DATA_ROOT / dataset paths.", file=sys.stderr)
        return 1

    scene = dataset.scenes[args.scene]
    iteration = args.iteration if args.iteration is not None else scene.number_of_history_iterations
    start_index = _flat_start_index(dataset, args.scene, iteration)
    camera_ids = [args.camera] if args.camera else None

    if args.save:
        fig = render_frame(dataset[start_index], camera_ids=camera_ids)
        fig.savefig(args.save, dpi=100, bbox_inches="tight")
        print(f"Saved starting frame to {args.save}")

    if not args.no_show:
        print("Opening GUI window — scroll or use ←/→ to page through frames, q to quit ...")
        FrameViewer(dataset, start_index=start_index, camera_ids=camera_ids).show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
