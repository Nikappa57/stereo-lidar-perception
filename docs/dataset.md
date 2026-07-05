# Dataset Choice — Which py123d Dataset for Stereo + LiDAR Perception

This project fuses a **stereo-camera branch** (SGBM disparity → metric depth → BEV,
see [perception_pipeline.md](perception_pipeline.md) §4–5) with a **LiDAR branch**
(PointPillars) in a shared BEV grid. The dataset therefore has a **hard requirement**
that most autonomous-driving datasets do *not* satisfy.

---

## 1. The hard requirement: a real stereo pair

The `StereoBEV` / SGBM branch needs a **genuine forward stereo pair** — two cameras
with:

- large **overlapping** field of view (same scene from both),
- (near-)parallel optical axes so the pair is **rectifiable** (epipolar lines →
  image rows, 1-D disparity search),
- a **known baseline** (metres) to turn disparity into metric depth
  (`depth = fx · baseline / disparity`).

A surround-view rig (front / front-left / front-right / …) does **not** qualify:
adjacent cameras point in different directions with little overlap, so there is no
rectifiable pair and no usable disparity. That single requirement eliminates most of
the catalogue.

A second, softer requirement for *this* project: the stereo pair should be **colour**
(RGB). The whole point of the camera branch is to splat *semantic* RGB features to
BEV; the geometric stereo BEV also carries mean-RGB channels
([perception_pipeline.md](perception_pipeline.md) §4.4). Grayscale still works
geometrically, but throws away colour signal.

---

## 2. Comparison of all py123d datasets

Filter columns first (`Stereo pair` and `Stereo colour`), then the usual perception
modalities.

| Dataset | Stereo pair? | Stereo colour? | Camera rig | LiDAR | 3D boxes | Auto-download | Volume |
|---|---|---|---|---|---|---|---|
| **KITTI-360** | ✅ `image_00/01` | ✅ **RGB** | 2 perspective (stereo) + 2 fisheye | Velodyne HDL-64 (~114k pts) | ✅ (global, static+dynamic) | ❌ login-gated | 9 long sequences |
| **Argoverse 2 (AV2)** | ✅ `pcam_stereo_l/r` | ❌ **grayscale** | 9 (7 ring + 2 stereo) | 2× VLP-32 merged (~93k pts) | ✅ human, dense (~23/frame) | ✅ `py123d-download` (public S3) | 1000 logs ×20 s |
| **CARLA** | ✅ configurable | ✅ (synthetic) | any (you place the rig) | ✅ | ✅ | ❌ must generate (LEAD) | synthetic, unlimited |
| nuScenes | ❌ surround | — | 6 surround (RGB) | 1× 32-beam | ✅ human | ✅ (needs account) | 1000 logs ×20 s |
| Waymo (WOD-Perception) | ❌ surround | — | 5 (RGB) | 5 | ✅ human | ✅ (GCS auth) | 1150 logs |
| nuPlan | ❌ surround | — | 8 surround (subset) | 5 (subset) | ✅ auto-labeled | ✅ (public S3) | ~1282 h |
| PandaSet | ❌ surround | — | 6 surround (RGB) | 2 | ✅ | ✅ (HF mirror) | 103 logs ×8 s |
| Physical-AI-AV | ❌ (7 fisheye) | — | 7 fisheye | 1× 360° | ✅ auto-labeled | ✅ (HF) | experimental |
| NCore | ❌ (7 fisheye) | — | 7 fisheye | 1× 360° | ✅ auto-labeled | ✅ (HF, gated) | experimental |
| WOD-Motion | ❌ (no raw sensors) | — | — | — | (agents only) | — | motion-only |

Only **three** rows clear the stereo-pair bar: KITTI-360, AV2, CARLA. Everything
below the divider is surround/fisheye — usable for a LiDAR-only or monocular-BEV
baseline, but **not** for the stereo branch.

---

## 3. Per-candidate verdict

**AV2 (the original choice).** Real stereo pair, public one-command download, dense
human-labeled boxes (~23/frame), 1000 logs. **But the stereo pair is grayscale** —
verified directly on the local data: R=G=B exactly (mean channel-diff `0.000`), while
the ring cameras *are* colour but are not a stereo pair. Great for the LiDAR branch and
for a grayscale-stereo baseline; does not meet the colour goal.

**KITTI-360.** The only ready-to-use dataset with a **real colour stereo pair**.
`image_00`/`image_01` map to py123d's `PCAM_STEREO_L`/`PCAM_STEREO_R`, so the existing
dataset-agnostic loader ([data.md](data.md) §1) works unchanged. Velodyne HDL-64 is
denser than AV2's merged cloud. Boxes are annotated in global coordinates with
visibility filtered by LiDAR-point count — the project's global→ego convention already
handles that. Downsides: **login-gated** (no `py123d-download`), only **9 sequences**
(less scene diversity), and **sparser boxes** (~5/frame vs AV2's ~23).

**CARLA.** Synthetic; you can place a virtual colour stereo rig with any baseline and
get perfect depth GT — ideal for augmentation / sim2real. Cost: you must *generate* the
data (LEAD framework), and there is a domain gap. Optional add-on, not a primary.

---

## 4. Recommendation

> **Primary: KITTI-360.** It is the only py123d dataset that delivers a real
> **colour** stereo pair together with LiDAR and 3D boxes — exactly the modalities this
> pipeline consumes. Adopt it as the main dataset for the stereo-fusion experiments.

Supporting roles:

- **AV2** — keep as a secondary/baseline source: many logs, dense boxes, one-command
  download. Use it to pretrain / benchmark the **LiDAR branch** and for a
  grayscale-stereo baseline. Do **not** mix AV2 frames into a "colour stereo" training
  set — grayscale frames would dilute the colour signal.
- **CARLA** — optional synthetic augmentation if KITTI-360's 9 sequences prove too few.

### Verified on KITTI-360 `drive_0003` (via `to_stereo_sample`)

| Property | KITTI-360 | AV2 (previous) |
|---|---|---|
| Stereo colour | ✅ COLOR (chan-diff ≈ 18.5) | ❌ grayscale (chan-diff 0.0) |
| Rectified resolution | 376 × 1408 | 1550 × 2048 |
| Baseline | 0.594 m | 0.4996 m |
| Left `fx` | 552.6 | 1688 |
| LiDAR points/frame | ~114k | ~93k |
| 3D boxes/frame | ~5.3 (max 29) | ~23 |
| Box labels seen | VEHICLE, GENERIC_OBJECT, TRAFFIC_SIGN | REGULAR_VEHICLE, … |

**Net trade-off:** gain a colour stereo pair and a denser LiDAR sweep; give up box
density and log diversity. For a stereo-centric pipeline that is the right trade.

---

## 5. Getting the data

Use [`scripts/get_kitti360.sh`](../scripts/get_kitti360.sh) — it downloads the public
per-sequence image/LiDAR/timestamps from KITTI-360's S3 bucket, extracts them into the
layout py123d expects, and runs the conversion into `data/logs/kitti360_{train,val}/`
ready for training. The small **login-gated** files (`calibration/`, `data_poses/`,
`data_3d_bboxes/`) must be downloaded once manually from the
[KITTI-360 download page](https://www.cvlibs.net/datasets/kitti-360/) and placed under
`$KITTI360_DATA_ROOT/` — the script checks for them and tells you if they are missing.
