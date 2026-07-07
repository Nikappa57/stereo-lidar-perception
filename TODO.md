# TODO — Stereo + LiDAR BEV Fusion

Output: **2D BEV `(x, y) + class`** (no yaw/z). Metric: center-distance
**AP @0.5/1/2/4 m**. Dataset: **KITTI-360**, **4-class**
(`VEHICLE / PERSON / TWO_WHEELER / TRAFFIC_SIGN`; TRAIN dropped — 0 GT on the
train drives). Split by drive: train `0003+0007+0009`, val `0010`.
Only **VEHICLE** and **TRAFFIC_SIGN** have enough GT to learn/eval;
PERSON/TWO_WHEELER are sparse — compare **per-class AP**, not just mAP.

---

TEST efficient network instead of yolo26

## ▶ Now — collect the numbers

- [ ] Train **camera** baseline + **pipeline_a / b / c** (`notebooks/training.ipynb`, same split/seed).
- [ ] Run **Pipeline D** (§8b) and fill the **`confronto.ipynb`** table (baselines vs fusion).

## Camera-branch experiments (active)

Test each modification in isolation (one variable, fixed seed/split) and read
the delta from the auto-saved run dirs (`runs/<name>_<ts>/`, compared in §9).
Data is tiny + overfitting-dominated, so single-run gaps are often noise — use
≥2–3 seeds and judge on **per-class AP** (VEHICLE / TRAFFIC_SIGN). Fold outcomes
back into [`docs/experiments.md`](docs/experiments.md).

**Phase 0 — measurement rig** (do before trusting any comparison)
- [x] **Run-dir autosave** (ultralytics-style) — `train.create_run` + `train_model(run_dir=…)` write `weights/{best,last}.pt`, `metrics.csv` (loss + P/R/F1/mAP per epoch), `results.json`, `summary.txt`, `plots/`, `config.json` (+ git SHA); `utils.save_eval_artifacts`.
- [x] **Decoder NMS** — greedy within-class **metric radius** removes duplicate peaks on large objects (a car holds several the 3×3 max-pool misses). Knob `NMS_RADIUS_BY_CLASS`; default off. A/B: `{}` vs `{0:2.0,1:0.5,2:1.0,3:0.5}`.
- [x] **Patch-based split** — `train.split_patches(dataset, val_frac, patch_len, seed, gap)` chops each drive into contiguous `patch_len` patches and assigns whole patches per-scene (less leakage than per-frame — only patch boundaries straddle the split, and `gap` buffers even those; more balanced than whole-log — every drive feeds both sides). Notebook knob `SPLIT_MODE="patch"` (+ `PATCH_LEN/PATCH_VAL_FRAC/PATCH_GAP`, snapshotted into `config.json`). **Drive 0010 (kitti360_val) stays an untouched test set** — never patched; evaluate on it separately.

**Phase 1 — attack overfitting** (the actual bottleneck: best-val ~epoch 2)
- [ ] **Weight decay** (AdamW) + **BEV augmentation** (rotation/flip/scale) — see *Training quality* below.
- [ ] **Dropout** in the head / BEV backbone.

**Phase 2 — feature & capacity upgrades** (after Phase 1 + more data)
- [ ] **YOLO neck P3+P4(+P5)** — concat multi-scale image features (the hook already captures all three); orthogonal, applies to every camera variant.
- [ ] **Depth → context head** — concat depth + validity mask into the image context features at 1/8 res (image-space; **no BEV projection needed**). With SGBM/IGEV depth = "features see geometry"; with LiDAR depth = Pipeline B at feature level.

**Phase 3 — diagnostics / deferred**
- [ ] **LiDAR range-map instead of stereo depth** — diagnostic *upper bound* on what better depth buys (uses LiDAR → not camera-only; not a deployment arm).
- [ ] **Loss tweaks** — class weighting for imbalance, Gaussian-radius / offset-weight tuning; defer until a specific symptom appears.

## Eval extensions

- [ ] **CDS** composite metric.
- [ ] **Per-range AP bins** (0–15 / 15–30 / 30–50 m).
- [ ] Orin **latency / FPS**.

## Training quality

- [ ] **BEV augmentation** (rotation / flip / scale, GT sampling).
- [ ] **Warm-start** the fused runs from the trained baselines.
- [ ] **Beam-downsampling** util (64→32→16) — unlocks the density ablation.

## Ablations (P3)

## P1 — Single-sensor baselines + eval harness (~weeks 2–3)

**Gate:** two baselines with numbers + an automated benchmark report.

- [x] **`train.py`** — training loop: `train_model` (per-log train/val split, batch-1 + gradient accumulation, shuffling, best-val checkpoint) over the CenterPoint loss (focal heatmap + masked L1 offset; classification implicit per-channel). Stereo input cache (`data.precompute_stereo_inputs` → `Pipeline(stereo_cache_root=…)`) removes SGBM from the step cost. Still TODO: BEV augmentation (rotation/flip/scale, GT sampling), warm-start, batched branches for throughput.
- [ ] **Camera-only BEV** baseline (stereo-depth splat → head).
- [ ] **LiDAR-only BEV** baseline (pillars → head) — run via `notebooks/training.ipynb` (`MODEL="lidar"`) on the 4/1-log split; numbers pending.
- [x] **`evaluation.py`** — AV2 **distance-AP** @0.5/1/2/4 m per class + mAP + mean centre error (`evaluate_model` / `print_ap_report`). Still TODO: CDS, per-range bins (0–15/15–30/30–50 m), Orin latency/FPS.
- [ ] **Training notebook** (`notebooks/training.ipynb`) — imports/globals/data/cache/network/train/test blocks calling the modules; used for the baseline runs.

---

## P2 — Pipeline A: CNN mid fusion (~weeks 4–6)

**Gate:** A beats both baselines, real-time on Orin. Minimum viable result.

- [ ] Assemble end-to-end: branches → `ConcatConvFusion` → `CenterPointHead`.
- [ ] Fine-tune at low LR; optionally freeze the camera backbone for the first epochs, then unfreeze (branches co-adapt — don't freeze permanently).

---

## P3 — Ablations (~weeks 6–8)

**Gate:** the ablation table, with the splat and beam-density rows quantified. Change one variable at a time; report mean ± std over ≥3 seeds.

> Full run plan (camera-branch × fusion matrix, run order, tags, baseline
> numbers): [`docs/experiments.md`](docs/experiments.md).

- [ ] **Stereo-splat vs predicted-depth** (StereoBEV vs MonoBEV) — *the headline experiment*; proves grounded stereo depth earns its place.
- [ ] Painted-range on/off (Pipeline B).
- [ ] Beam density 64 / 32 / 16 (AV2 ↔ VLP-16 procurement argument).
- [ ] Multi-sweep accumulation on/off.
- [ ] **Cone transferability** — TRAFFIC_SIGN / small-object breakout as the FS cone proxy.
- [ ] Pipeline C: init from the trained A, **range-stratified A-vs-C** verdict.

## Deployment & write-up (P5)

- [ ] Paper + failure analysis (far / small / occluded qualitative panel).
- [ ] ZED stereo front-end; calibrate ZED ↔ VLP-16; re-tune the near/far crossover for 16 beams.
- [ ] Auto-label FS cone data, retrain (method transfers, weights don't); TensorRT export on the Orin.

---

## Open decisions (settle as a team)

- BEV grid extent & resolution vs latency (current 50 m / 0.25 m).
- Attention scope for Pipeline C (currently windowed 8×8).
- Scope line: stop at the CNN-vs-transformer verdict (P4), or push painted-range + multi-sweep to completion.
