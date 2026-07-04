# BEV Fusion & Detection Head (`network.py`)

This document describes the **BEV fusion** block and the 2D detection head — the *consumer* side of the mid-fusion pipeline. It explains exactly what the block needs as input, why, and how those inputs were chosen.

We build this block **first**, on purpose. Defining the consumer fixes the **contract** for Stage A: once we know what the fusion expects, we know what the two branches (camera, LiDAR) must output. The flow is therefore *fusion → read the contract → go back and confirm Stage A emits it*.

---

## 1. Core principle (why fusion needs aligned grids)

> Two feature maps may be combined cell-by-cell **only if their cells refer to the same physical place.** (Design doc §02.)

The shared **ego bird's-eye-view grid** is that common place: the only frame both sensors can reach by known geometry, and the only one the 2D `(x, y) + class` output needs. So BEV fusion requires its two inputs to be **pixel-aligned on the same grid** — same `(nx, ny)`, same metric extent, same cell size.

`BEVFusion.forward` enforces this with a shape guard (it raises if the grids or channel counts disagree), turning §02 into an executable invariant: a future change in Stage A cannot silently feed misaligned maps into fusion.

---

## 2. What BEV fusion needs (the contract)

Exactly **two** BEV feature maps, both on the shared grid:

| Input | Shape | Produced by |
| --- | --- | --- |
| `bev_camera` | `(B, 64, 200, 160)` | Camera branch — grounded stereo-depth splat (`StereoBEVBranch`) |
| `bev_lidar` | `(B, 128, 200, 160)` | LiDAR branch — PointPillars (`PointPillarsBranch`) |
| **fused output** | `(B, 128, 200, 160)` | `ConcatConvFusion` → fed to the head |
| **head output** | `heatmap (B, 3, 200, 160)` + `offset (B, 2, 200, 160)` | `CenterPointHead` |

Why **two** maps (not more): this is mid-fusion — the camera and LiDAR branches stay separate and meet **once**, in BEV. Pipeline B does not add a third fusion input; its painted LiDAR-range channel is injected **upstream**, inside Stage A, so the BEV fusion block still sees exactly two maps.

Why **concatenate** (not add): concatenation keeps every channel of both sensors and lets the convolution learn how to mix them. Element-wise `add` would require equal channel counts (64 ≠ 128) and assume comparable feature scales. Concat is robust to the camera/LiDAR channel asymmetry.

---

## 3. How the dimensions were chosen

The key point: **the fusion does not invent its input channels — it reads whatever Stage A already emits.** `BEVDetector.from_bev_maps(bev_camera, bev_lidar)` derives the channel counts and grid directly from the two branch tensors, so the contract can never drift from the branches.

### 3.1. Grid `(nx, ny) = (200, 160)`

Derived from the shared BEV extent and resolution used by every branch (`PillarConfig`, `StereoBEVConfig`, `MonoBEVConfig`):

- `x ∈ [0, 50] m` (forward), `y ∈ [-20, 20] m` (lateral), `0.25 m / cell`
- `nx = 50 / 0.25 = 200`, `ny = 40 / 0.25 = 160`

The exact extent/resolution is a tunable choice (design doc §14 lists it as an open decision — AV2 labels reach 150 m+), but it **must** be identical across both branches and the fusion.

### 3.2. Camera channels `= 64`

This is `StereoBEVConfig.context_channels` — the depth of the **learned semantic feature** vector that the camera branch splats onto BEV (not geometric values). 64 is a moderate feature width: enough capacity while staying light for the Orin.

### 3.3. LiDAR channels `= 128`

PointPillars runs `PillarFeatureNet` (64) → scatter → `BEVBackbone2D(out=128)`, so after the BEV backbone the LiDAR map is 128 channels — the standard pillar-BEV width.

### 3.4. Fusion-side choices (tunable)

- `out_channels = 128` — depth of the fused feature map fed to the head (≈ the LiDAR width). A design choice, easily tuned.
- `head_channels = 64` — width of the head's shared conv stem.
- `num_classes = 3` — placeholder for `REGULAR_VEHICLE + CONSTRUCTION_CONE + one small class` (design doc §05). Fixed once the class filter (a separate TODO) pins the final set.

Reference sizes: fusion ≈ 369 k params, head ≈ 74 k params (≈ 443 k total).

---

## 4. The detection head (output)

`CenterPointHead` reads object centres off the fused map and produces **2D only** targets (design doc §08 — no yaw, no z; our objects are ground-planar):

- `heatmap (B, num_classes, nx, ny)` — per-class centre logits (sigmoid in the focal loss / at inference). The heatmap bias is initialised to `-2.19` (focal prior) so training is not swamped by the overwhelmingly empty grid.
- `offset (B, 2, nx, ny)` — the sub-cell `(dx, dy)` of the centre within its cell.

---

## 5. Swappable fusion (same interface across pipelines)

The fusion block has a **fixed interface** — *two BEV maps in, one out* — so it is swapped per pipeline without touching the branches or the head:

| Pipeline | Fusion block | Difference vs A |
| --- | --- | --- |
| **A** | `ConcatConvFusion` (concat + conv) | — |
| **B** | `ConcatConvFusion` (**identical to A**) | upstream only: painted LiDAR-range channel inside Stage A |
| **C** | `CrossAttentionFusion` (near→stereo, far→LiDAR) | swaps the fusion block; same interface, drop-in |
| **D** | **none** | late fusion merges object lists, not feature maps — does not use this module |

Both `ConcatConvFusion` and `CrossAttentionFusion` subclass the abstract `BEVFusion`, so moving from the CNN baseline (A) to the cross-attention novelty (C) is a one-class change.

---

## 6. Usage

```python
from network import BEVDetector, describe

# Build the detector straight from the Stage A branch outputs:
detector = BEVDetector.from_bev_maps(bev_camera, bev_lidar, num_classes=3).eval()

describe(detector)                 # prints the contract Stage A must satisfy + params
out = detector(bev_camera, bev_lidar)
# out["heatmap"] -> (B, num_classes, nx, ny)
# out["offset"]  -> (B, 2, nx, ny)
```

`describe()` prints the contract; the branches already emit `64` / `128` on grid `(200, 160)`, so Stage A is **already aligned** — no change needed there.

---

## 7. What's next

The target side of this contract is now in place: `train.TargetEncoder` rasterises the ego-frame GT boxes (`StereoSample.boxes_3d_ego` centres + class) into the **same** `heatmap (num_classes, nx, ny)` + `offset (2, nx, ny)` tensors the head produces, `train.CenterPointLoss` scores them (Gaussian-focal heatmap + masked L1 offset), and `evaluation.CenterPointDecoder` inverts the head output back to metric ego `(x, y)` + class. The encode→decode round-trip is covered by `tests/test_encoder_decoder.py`, and `train.overfit_one_frame` drives the whole loop (encoder → head → loss → backward → decode) on one frame.

The remaining gap is the **multi-frame training loop + AP/CDS evaluation harness** (TODO P1): batch the branches over a real train split, then score the single-sensor baselines and the fused detector with distance-AP.
