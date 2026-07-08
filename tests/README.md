# tests/

## `test_data.py`

Tests and a visual check for the `data.py` loader. Two modes:

### `pytest tests/test_data.py` — headless verification
Asserts the dataset loads and every modality is well-formed:
scenes/frames load, camera images are `uint8 HxWx3` matching their metadata,
lidar is `(P,3)`, 3D boxes are valid, the ego pose is finite, the 3D boxes
**project into at least one camera** (intrinsics + extrinsics check), and the
assembled `StereoSample` is internally consistent (shapes, 2D↔3D mapping,
point-in-box split, ~0.5 m stereo baseline).

### `python tests/test_data.py` — interactive viewer
Opens a GUI window showing one full-size camera image with the 3D bounding boxes
drawn on top.

- scroll / `←→` — change frame
- `↑↓` — switch camera
- `f` fullscreen · `q` quit

Flags: `--split`, `--scene`, `--iteration`, `--camera pcam_f0`, `--max-scenes N`,
`--save PATH` (saves a 9-camera grid), `--no-show`.

Paths default to the repo `data/` dir, so no env vars are needed.

## Other test files

Each has the same two-mode pattern (`pytest` = headless assertions, `python
tests/<file>.py` = a runnable visual check saved under `docs/img/`):

- `test_stereo.py` — SGBM rectification/disparity/depth + stereo-vs-LiDAR
  accuracy check (`data.py`).
- `test_network.py` — BEV fusion + head shape/contract guards, random-tensor
  fast tests, plus a real-frame fused-BEV figure (`network.py`).
- `test_encoder_decoder.py` — GT boxes survive the encode→decode round-trip
  (`train.TargetEncoder` ↔ `evaluation.CenterPointDecoder`).
- `test_overfit.py` — end-to-end sanity: overfit one real frame
  (LiDAR-only / Pipeline A / Pipeline C) and recover the GT centres.
