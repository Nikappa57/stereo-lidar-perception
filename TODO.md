# TODO — Stereo + LiDAR BEV Fusion

Working list for the Pipeline A build, phased to match the design document
(§ references point to it). Output target is **2D BEV `(x, y) + class`** — no
yaw, no z (§08). Metric is AV2 **distance-AP @0.5/1/2/4 m + CDS** (§05).

Legend: `critical` / `high` / `medium` / `low` = §11 priority.

---

## ✅ Done

- **Data** (`data.py`): py123d loader → per-frame `StereoSample`; 2D boxes auto-projected from 3D; calibration/extrinsics exposed.
- **Frame fix** (`data.py`): global→ego box conversion (`boxes_3d_ego`) + frame-consistency guard (`assert_boxes_in_sensor_range`). — §11 critical (half) + medium.
- **Preprocessing** (`preprocessing.py`): BEV / voxel / frustum / DBSCAN representations.
- **Stereo depth** (`stereo.py`): SGBM rectification + disparity → metric depth, wired into the stereo branch. — §11 critical (SGBM path; RAFT-Stereo still optional, §14).
- **Camera→BEV splat**: `MonoBEV` (predicted-depth LSS) and `StereoBEV` (grounded stereo-depth splat). — §11 critical.
- **LiDAR BEV**: `PointPillars` branch → `(128, 200, 160)`.
- **BEV fusion + head** (`network.py`): `ConcatConvFusion` (A/B) + `CrossAttentionFusion` stub (C) behind a fixed `BEVFusion` interface; `CenterPointHead` (heatmap + offset, 2D only); `BEVDetector`. See [`docs/bev_fusion.md`](docs/bev_fusion.md).
- **Notebooks**: `pipeline_a.ipynb`, `pipeline_b.ipynb` (imports/globals/utils/data/network/train/test, calling the `.py` modules).

---

## P0 — Close the code gaps (~weeks 1–2, §12)

**Gate:** one frame yields aligned BEV inputs (camera + LiDAR) **and** a BEV training target.

- [ ] `critical` **BEV target encoder** — rasterise `boxes_3d_ego[:, :2]` (centres) + class into the **same** tensors the head emits: `heatmap (num_classes, 200, 160)` + `offset (2, 200, 160)`. Gaussian splat per centre, sub-cell offset, class index. *This is the last thing missing before any training.*
- [ ] `high` **Class selection & remap** — filter the 26–30 AV2 categories to the working subset (`REGULAR_VEHICLE` + `CONSTRUCTION_CONE` + one small class) and define a fixed label map; plan the AV2→FS cone remap for deployment. Sets `num_classes`.
- [ ] `high` **Beam-downsampling util** — use the per-point channel in `lidar_features` to drop beams (64→32→16). One small function; unlocks the §08 density ablation.
- [ ] `medium` **`globals.py`** — consolidate the shared grid/config (x/y range, resolution, class map) into one source of truth (currently duplicated across `PillarConfig` / `StereoBEVConfig` / `MonoBEVConfig`). Also rename to avoid shadowing the builtin `globals()`.
- [ ] `low` **Confirm stereo timing** — AV2 stereo runs slower than LiDAR; verify the sync the loader returns and that left/right are the same exposure.
- [ ] **Pipeline B wiring** — inject the painted-range channel (`depth_left`) into the Stage A image fusion, behind a toggle (sparsity-aware conv). The BEV fusion block is unchanged.

---

## P1 — Single-sensor baselines + eval harness (~weeks 2–3)

**Gate:** two baselines with numbers + an automated benchmark report.

- [ ] **`train.py`** — training loop. CenterPoint loss: focal heatmap + L1 `(x, y)` offset + CE class. BEV augmentation (rotation/flip/scale, GT sampling). Warm-start support.
- [ ] **Camera-only BEV** baseline (stereo-depth splat → head).
- [ ] **LiDAR-only BEV** baseline (pillars → head).
- [ ] **`evaluation.py`** — AV2 distance-AP/CDS, per-range bins (0–15 / 15–30 / 30–50 / >50 m), per-class (cones reported separately), mean `(x, y)` error in cm, Orin latency/FPS/params.

---

## P2 — Pipeline A: CNN mid fusion (~weeks 4–6)

**Gate:** A beats both baselines, real-time on Orin. Minimum viable result.

- [ ] Assemble end-to-end: branches → `ConcatConvFusion` → `CenterPointHead`.
- [ ] Fine-tune at low LR; optionally freeze the camera backbone for the first epochs, then unfreeze (branches co-adapt — don't freeze permanently).

---

## P3 — Ablations (~weeks 6–8)

**Gate:** the ablation table, with the splat and beam-density rows quantified. Change one variable at a time; report mean ± std over ≥3 seeds.

- [ ] **Stereo-splat vs predicted-depth** (StereoBEV vs MonoBEV) — *the headline experiment*; proves grounded stereo depth earns its place.
- [ ] Painted-range on/off (Pipeline B).
- [ ] Beam density 64 / 32 / 16 — bridges the AV2 ↔ VLP-16 gap; procurement argument.
- [ ] Backbone 3D-then-collapse vs early-BEV (use the `voxel_grid` path).
- [ ] Multi-sweep accumulation on/off.
- [ ] Cone-class breakout (transfer relevance to Formula Student).

---

## P4 — Pipeline C: cross-attention (~weeks 8–10)

**Gate:** CNN-vs-transformer verdict with per-range numbers + overhead.

- [ ] Implement `CrossAttentionFusion` (deformable / windowed, or candidate-cell queries — full grid is quadratic). Drop-in via the existing `BEVFusion` interface.
- [ ] Initialise from the trained Pipeline A model; range-stratified comparison.

---

## P5 — Write-up & FS deployment (~weeks 10–12)

**Gate:** submitted study + a fusion module running in the live car pipeline.

- [ ] Paper + failure analysis (qualitative panel: far/small/occluded).
- [ ] Swap stereo front-end to ZED hardware depth; calibrate ZED ↔ VLP-16.
- [ ] Re-tune the near/far crossover for the 16-beam VLP-16.
- [ ] Auto-label FS cone data; retrain (method transfers, weights don't).
- [ ] TensorRT export; integrate on the Orin.

---

## Open decisions (§14, settle as a team)

- Stereo matcher: SGM (current) vs RAFT-Stereo / learned.
- Class subset size (cones in either way).
- BEV grid extent & resolution vs latency (AV2 reaches 150 m+; current default 50 m / 0.25 m).
- Attention scope for Pipeline C (candidate-cell vs windowed full-grid).
- Scope line: stop at the CNN-vs-transformer verdict (P4), or push painted-range + multi-sweep to completion.
