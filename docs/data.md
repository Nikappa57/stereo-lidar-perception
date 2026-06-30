# Data Component (`data.py`) — Reference

Documentation for the **data loading** stage of `stereo-lidar-perception`, plus
everything we verified about the `py123d` dataset library while building it.

All facts below were confirmed against **`py123d` 0.5.1** and the local
Argoverse 2 Sensor sample in `data/` (5 validation logs).

---

## 1. Pipeline context

```
raw dataset ─► py123d (unified Arrow format) ─► data.py ─► preprocessing ─► network ─► train/eval
```

`data.py` is the *data* stage. It reads raw sensors, calibration and labels
through the unified `py123d` **Scene API** and assembles a per-frame
**`StereoSample`**, then provides the geometric preprocessing representations
(stereo depth/BEV, frustum, voxel, clustering) built from it.

Because everything goes through `py123d.api.SceneAPI`, the loader is
**dataset-agnostic**: switching from Argoverse 2 to nuPlan / nuScenes / Waymo /
… only changes the `SceneFilter` and the root paths. The stereo-specific
assembly assumes a left/right camera pair whose ids are configurable
(AV2's `pcam_stereo_l` / `pcam_stereo_r` by default).

---

## 2. On-disk data layout

`py123d` converts each dataset into a common format. Three roots matter:

```
data/
├── logs/                                   # $PY123D_DATA_ROOT/logs  (converted, Arrow)
│   └── av2-sensor_val/
│       └── <log-uuid>/
│           ├── sync.arrow                   # modality synchronization table
│           ├── ego_state_se3.arrow
│           ├── box_detections_se3.arrow     # 3D box labels
│           ├── camera.pcam_f0.arrow … (×9)  # one per camera (relative img paths)
│           ├── lidar.lidar_merged.arrow
│           └── map.arrow                     # per-log HD map (AV2)
├── sensor/                                  # $AV2_DATA_ROOT/sensor  (original blobs)
│   └── val/<log-uuid>/
│       ├── annotations.feather
│       ├── calibration/{egovehicle_SE3_sensor,intrinsics}.feather
│       ├── city_SE3_egovehicle.feather
│       ├── map/…
│       └── sensors/{cameras,lidar}/…        # the actual JPEGs / lidar feathers
└── preprocessed/
    └── depth_maps/<log-uuid>/<iter:05d>.npz # precomputed sparse depth (see §8)
```

> **Why two roots?** AV2 conversion stores only *relative paths* to images/lidar
> in the `.arrow` files (not the pixels). At read time the camera/lidar readers
> resolve those against the **sensor root**. Streaming conversion
> (`av2-sensor-stream`) instead embeds the blobs, so it needs only
> `$PY123D_DATA_ROOT`.

---

## 3. Path configuration

`py123d` resolves all roots from **environment variables** (via
`DatasetPaths.from_env()`; the hydra `default_dataset_paths.yaml` uses
`${oc.env:…}`):

| Env var | Resolves to | Used for |
|---|---|---|
| `PY123D_DATA_ROOT` | `<root>/logs`, `<root>/maps` | converted scenes |
| `AV2_DATA_ROOT` | `<root>/sensor` | AV2 camera/lidar blobs |
| `NUPLAN_DATA_ROOT`, `NUSCENES_DATA_ROOT`, … | per-dataset | other datasets' sensors |

In this repo the data is self-contained, so **both** roots point at `data/`.

`data.py` wires this up in `configure_dataset_paths(data_root)`:

- data root = arg → `$PY123D_DATA_ROOT` → repo `data/` (in that order);
- already-exported dataset vars are kept, the rest default to the data root;
- it must run **before** any scene/sensor read (the readers look paths up through
  a process-global, lazily-cached `DatasetPaths`). `Py123dDataset` calls it
  automatically.

**Gotcha:** forget `AV2_DATA_ROOT` and camera reads raise
`AssertionError: Dataset path for sensor loading not found for dataset: av2-sensor`.

---

## 4. Loading scenes (the py123d entry point)

```python
from py123d.api import SceneFilter, get_filtered_scenes
scenes = get_filtered_scenes(
    SceneFilter(split_names=["av2-sensor_val"], max_num_scenes=2),
    data_root=Path("data").resolve(),       # logs/maps root
)
```

`SceneAPI` is an abstract base — you never construct it directly. `SceneFilter`
selects by `datasets`, `split_names`, `log_names`, `scene_uuids`,
`max_num_scenes`, `required_scene_modalities`, `shuffle`, history/future
windows, etc. There is also a `ViserViewer` CLI builder path (see §10).

A scene exposes, among others:

| Attribute / method | Notes (local AV2 sample) |
|---|---|
| `dataset`, `split`, `location` | `av2-sensor`, `av2-sensor_val`, `PIT` |
| `log_name` | log directory uuid (`02678d04-…`) |
| `scene_uuid` | a *different* uuid from `log_name` |
| `number_of_iterations` | 157 (current + future frames) |
| `number_of_history_iterations` | 0 (the "current" frame index) |
| `available_camera_ids` | 9 `CameraID` enums |
| `available_camera_names` | human names (`ring_front_center`, …) |
| `available_lidar_ids` | `LIDAR_TOP`, `LIDAR_DOWN`, `LIDAR_MERGED` |
| `get_camera_at_iteration(it, camera_id)` | → `Camera` |
| `get_lidar_at_iteration(it, lidar_id)` | → `Lidar` |
| `get_box_detections_se3_at_iteration(it)` | → `BoxDetectionsSE3` |
| `get_ego_state_se3_at_iteration(it)` | → `EgoStateSE3` |

> **Gotcha:** camera/lidar getters want a `CameraID`/`LidarID` **enum or its
> lowercase enum-name** (`"pcam_f0"`). Passing the human name
> (`"ring_front_center"`) raises `KeyError: 'RING_FRONT_CENTER'`. Iterate
> `available_camera_ids`, not `available_camera_names`.

---

## 5. Modality details (verified)

### Cameras (9)
`PCAM_F0, PCAM_L0, PCAM_L1, PCAM_L2, PCAM_R0, PCAM_R1, PCAM_R2, PCAM_STEREO_L,
PCAM_STEREO_R`.

Resolutions (H × W): `PCAM_F0` is **2048 × 1550** (portrait); all others,
including the stereo pair, are **1550 × 2048**.

`Camera`: `.image` (HxWx3 uint8), `.camera_to_global_se3`, `.project_points_global(pts_global)`,
`.timestamp`, `.metadata`. `Camera.metadata`: `.intrinsics` (`PinholeIntrinsics`,
5 values `[fx, fy, cx, cy, skew]`), `.camera_to_imu_se3` (cam→ego), `.width`,
`.height`, `.fov_x/.fov_y`, `.is_distorted`, `.project_to_image(pts_cam)`.

Stereo-left intrinsics: `fx = fy = 1688.18`, `cx = 1022.44`, `cy = 772.10`.

### LiDAR
`Lidar`: `.xyz` (P×3, **ego frame**), `.xy`, `.intensity`, `.range`,
`.elongation`, `.channel`, `.ids`, `.timestamps`, `.point_cloud_features`
(dict: `ids`, `intensity`, `channel`, `timestamps`), `.is_merged`. Merged cloud
≈ **93,509** points.

### 3D boxes
`BoxDetectionsSE3.box_detections` is a list of `BoxDetectionSE3`:
`.bounding_box_se3.array` (length-10, **global frame**), `.bounding_box_se2`
(BEV — *not* image-plane 2D), `.center_se3`, `.velocity_3d`, `.attributes`
(`default_label`, `label`, `num_lidar_points`, `track_token`). ~23 boxes at the
first frame.

`BoundingBoxSE3Index`: `X=0, Y=1, Z=2, QW=3, QX=4, QY=5, QZ=6, LENGTH=7,
WIDTH=8, HEIGHT=9`.

### Geometry helpers (`py123d.geometry`)
- `bbse3_array_to_corners_array(arr)` → `(N, 8, 3)` corners (global).
- `points_3d_in_bbse3_array(points (P,3), boxes (N,10))` → `(N, P)` bool mask.
- `rel_to_abs_points_3d_array(pose, pts)` → ego→global.
- `PoseSE3`: `.array` (7), `.rotation_matrix` (3×3), `.quaternion`, `.inverse`.

---

## 6. `data.py` public API

```python
from data import Py123dDataset, Frame, StereoSample, Calibration, configure_dataset_paths
```

### `Py123dDataset(data_root=None, *, split_names=None, datasets=None, depth_root=None, scene_filter=None, **filter_kwargs)`
Flat, frame-indexed view over the filtered scenes. One `(scene, iteration)` per
index → map-style `__len__` / `__getitem__` (drop-in for a
`torch.utils.data.DataLoader`). `depth_root` defaults to
`<data_root>/preprocessed/depth_maps`. Also: `.scenes`, `.scene_count`,
`.get_frame(scene, iter)`, `.frames_in_scene(scene)`.

### `Frame`
Lazy, dataset-agnostic accessors for one timestamp:
`camera(id)`, `cameras(ids=None)`, `lidar(id=None)` (defaults
merged→first), `boxes()`, `ego_state()`, `depth()`, plus metadata props
(`dataset`, `log_name`, `timestamp`, `available_camera_ids`, …). Any missing
modality returns `None`/empty so non-AV2 datasets work. Key method:
`to_stereo_sample(left_camera_id="pcam_stereo_l", right_camera_id="pcam_stereo_r", lidar_id=None) → StereoSample`.

---

## 7. `data.py` inputs / outputs

**Inputs:** converted logs (`$PY123D_DATA_ROOT/logs`), sensor blobs
(`$AV2_DATA_ROOT/sensor`), optional precomputed depth
(`preprocessed/depth_maps/<log>/<iter>.npz`), and a `SceneFilter`.

**Output — one `StereoSample` per frame:**

| Field | Type / shape | Source |
|---|---|---|
| `image_left`, `image_right` | `(H,W,3)` uint8 | raw stereo pair |
| `depth_left` | `(H,W)` float32 or `None` | precomputed sparse LiDAR depth (§8) |
| `lidar_xyz` | `(P,3)` ego frame | raw LiDAR |
| `lidar_features` | dict of `(P,)` channels | raw LiDAR |
| `boxes_3d` | `(N,10)` global frame | **ground-truth** labels |
| `boxes_3d_labels` / `_track_tokens` / `_velocity` | len-N / len-N / `(N,3)` | labels |
| `boxes_2d_left` | `(M,4)` xyxy, clipped | **auto-generated**: 3D boxes projected into left image |
| `boxes_2d_left_box_indices` | `(M,)` | maps each 2D box → row in `boxes_3d` |
| `points_outside_boxes_xyz` | `(Q,3)` ego frame | LiDAR returns outside *every* 3D box |
| `points_in_box_mask` | `(P,)` bool | membership over `lidar_xyz` |
| `calibration` | `Calibration` | intrinsics + extrinsics (§9) |

### Design notes / clarifications
- **3D boxes are dataset ground-truth** (human-annotated), not "auto-generated".
  The *2D* boxes are the auto-generated ones (projection of the 3D boxes).
- **Depth is not a raw py123d field** — it is loaded from the precomputed maps;
  `depth()` returns `None` cleanly when a file is absent, so the loader also
  works before preprocessing has run.
- **"3D position if not within the bounding boxes"** = LiDAR returns that fall
  outside all 3D boxes (static/background geometry), via
  `points_3d_in_bbse3_array`.

---

## 8. Depth: it is LiDAR, **not** stereo

The precomputed `depth_maps/<log>/<iter:05d>.npz` (key `depth_map`, shape
`(1550, 2048)` = stereo-frame resolution) is **sparse LiDAR depth projected into
the left stereo image**, not stereo/disparity depth. Verified:

- density **0.255 %** (≈8,097 lit pixels of 3.17 M) — LiDAR-sparse, not dense;
- re-projecting the merged LiDAR into `pcam_stereo_l` ourselves gives the **same
  max range (202.1 m)** and a **median depth difference of 0.000 m** on shared
  pixels — i.e. identical values.

**There is no stereo depth in the dataset.** The stereo camera provides only the
raw left/right image pair. Stereo depth must be **computed** (the project's
point):

```
depth = fx * baseline / disparity
```

with `fx = calibration.left_intrinsics[0,0]`,
`baseline = calibration.stereo_baseline_m` (≈ 0.5 m), and `disparity` from a
matcher/network (preprocessing or model — not loaded here). The LiDAR sparse
depth is the supervision / fusion signal.

---

## 9. Calibration

`Calibration` (metres, radians):

- `left_intrinsics`, `right_intrinsics` — `(3,3)` `K = [[fx,skew,cx],[0,fy,cy],[0,0,1]]`;
- `left_to_ego`, `right_to_ego` — `(4,4)` camera→ego (from `camera_to_imu_se3`);
- `ego_to_global` — `(4,4)` (from `EgoStateSE3.imu_se3`);
- `left_to_global`, `right_to_global` — `(4,4)`;
- `stereo_baseline_m` — `‖t_left − t_right‖`;
- `left_is_distorted`, `right_is_distorted`.

Verified: baseline = **0.4996 m**, matching AV2's known ~0.5 m stereo baseline
(left cam center `y=+0.258`, right `y=−0.242` in ego frame). Stereo cams are
undistorted (`is_distorted = False`).

---

## 10. Visualization / viewers

### a) Project viewer — `tests/test_data.py`
Dual purpose:
- `pytest tests/test_data.py` → **8 headless tests**: scenes load; cameras are
  uint8 HxWx3 matching metadata; lidar `(P,3)`; boxes well-formed; ego pose
  finite; **boxes project into ≥1 camera** (intrinsics+extrinsics check); and the
  full **`StereoSample` contract** (shapes/consistency, baseline sane).
- `python tests/test_data.py` → interactive **`FrameViewer`**: one full-size
  camera image at a time with 3D boxes overlaid.
  - scroll / `←→` / `n p` / `space` — change **frame** (wraps);
  - `↑↓` / `,` `.` — switch **camera** (wraps);
  - `f` fullscreen, `q` quit.
  - flags: `--split`, `--scene`, `--iteration`, `--camera pcam_f0`,
    `--max-scenes N`, `--save PATH` (9-up grid overview), `--no-show`.

py123d's matplotlib helpers used: `add_box_detections_to_camera_ax(ax, camera, boxes)`
(projects 3D boxes onto the image), `add_camera_ax`, `add_lidar_to_camera_ax`.

### b) py123d's built-in Viser web GUI — `py123d-viser`
3D web viewer (lidar + boxes + map + camera frustums), `viser 1.0.24` installed.

```bash
export PY123D_DATA_ROOT="$PWD/data"   # converted logs/maps
export AV2_DATA_ROOT="$PWD/data"      # raw sensor blobs ($AV2_DATA_ROOT/sensor)
py123d-viser scene_filter=av2-sensor \
  'scene_filter.split_names=[av2-sensor_val]' scene_filter.max_num_scenes=3
# open http://localhost:8080
```

Both env vars are required (logs *and* sensor blobs); the default filter loads
all datasets, so `scene_filter=av2-sensor` narrows it. Verified serving HTTP 200.

---

## 11. Quick start

```python
from data import Py123dDataset

dataset = Py123dDataset(split_names=["av2-sensor_val"])   # paths auto-configured
print(len(dataset))                                       # 157 frames in 1 scene

sample = dataset[13].to_stereo_sample()                   # -> StereoSample
sample.image_left            # (1550, 2048, 3) uint8
sample.depth_left            # (1550, 2048) float32 sparse, or None
sample.boxes_3d              # (N, 10) global-frame GT boxes
sample.boxes_2d_left         # (M, 4) image boxes from 3D projection
sample.calibration.stereo_baseline_m   # ~0.4996
```

---

## 12. Environment / versions

- `py123d` **0.5.1**, `viser` **1.0.24**, Python 3.10, pytest 9.0.3.
- Install: `pip install "py123d[av2]"` (the `[av2]` extra / `boto3` is only for
  *downloading*; parsing local data needs no extra).
- Local sample: 5 AV2 validation logs in `data/` (≈156–159 frames each), with
  matching precomputed depth maps.
