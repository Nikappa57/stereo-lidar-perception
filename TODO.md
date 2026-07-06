# TODO — Stereo + LiDAR BEV Fusion

Output: **2D BEV `(x, y) + class`** (no yaw/z). Metric: center-distance
**AP @0.5/1/2/4 m**. Dataset: **KITTI-360**, **5-class unified**
(`VEHICLE / PERSON / TWO_WHEELER / TRAFFIC_SIGN / TRAIN`). Split by drive:
train `0003+0007+0009`, val `0010`.

---

## ▶ Now — collect the numbers

- [ ] Train **camera** baseline + **pipeline_a / b / c** (`training.ipynb`, same split/seed).
- [ ] Run **Pipeline D** (§8b) and fill the **`confronto.ipynb`** table (baselines vs fusion).

## Eval extensions

- [ ] **CDS** composite metric.
- [ ] **Per-range AP bins** (0–15 / 15–30 / 30–50 m).
- [ ] Orin **latency / FPS**.

## Training quality

- [ ] **BEV augmentation** (rotation / flip / scale, GT sampling).
- [ ] **Warm-start** the fused runs from the trained baselines.
- [ ] **Beam-downsampling** util (64→32→16) — unlocks the density ablation.

## Ablations (P3)

- [ ] **Stereo-splat vs predicted-depth** (StereoBEV vs MonoBEV) — *the headline experiment*.
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
