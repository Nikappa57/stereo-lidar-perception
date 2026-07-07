# Training log


## CAMERA ONLY

### 1) baseline

```
print("grid:", G.GRID_SIZE, "| x:", G.X_RANGE, "| y:", G.Y_RANGE, "| classes:",
      G.CLASSES)

# --- run configuration -------------------------------------------------
MODEL = "camera"  # "lidar" | "camera" (baselines) | "pipeline_a" (fused)
EPOCHS = 5
LR = 1e-3
ACCUM = 4  # frames per optimizer step (batch-1 + accumulation)
VAL_SCENES = 1  # FALLBACK ONLY: used when kitti360_val isn't downloaded yet but
# kitti360_train has >1 log (holds out the last n). Ignored once the named
# kitti360_val split exists. int, or explicit log indices to hold out.
SEED = 0  # python/numpy/torch/CUDA RNGs (weight init, shuffling)

# camera stem (camera / pipeline_a only): "efficientnet" trains a ~1M-param CNN
# from scratch (weak); "yolo26" uses a COCO-pretrained backbone. FREEZE keeps it
# fixed so only the head/BEV/context train — the strong first baseline to try.
CAMERA_BACKBONE = "yolo26"  # "efficientnet" | "yolo26"
FREEZE_BACKBONE = True

tag = "" if MODEL == "lidar" else f"_{CAMERA_BACKBONE}"
CKPT = f"checkpoints/{MODEL}{tag}.pt"  # best-val weights (train_model writes it)
RESULTS = f"results/{MODEL}{tag}.json"  # eval report (§7 writes; §9 compares)

set_seed(SEED)  # reproducible runs (not bit-deterministic: CUDA atomics)
```

DATASET SPLIT
kitti360_train=0007 (2,890) / kitti360_val=0003 (1,010)


CameraOnlyDetector: 772,039 trainable | 2,572,280 frozen (yolo26 backbone)

21mitues

![alt text](docs/img/train/train0-loss.png)

class         AP@0.5  AP@1    AP@2    AP@4      mean   n_gt
-----------------------------------------------------------
VEHICLE       0.225   0.423   0.492   0.514   0.413  805
PERSON        0.000   0.000   0.000   0.000   0.000  18
TWO_WHEELER   —       —       —       —       —      0
TRAFFIC_SIGN  0.115   0.139   0.147   0.154   0.139  274
TRAIN         —       —       —       —       —      0

F1-optimal operating point @2 m (apply 'confidence >= score' at deployment):
class         prec    recall  F1      score   
----------------------------------------------
VEHICLE       0.497   0.579   0.535   0.218   
PERSON        0.000   0.000   0.000   nan     
TWO_WHEELER   —       —       —       —       
TRAFFIC_SIGN  0.338   0.245   0.284   0.204   
TRAIN         —       —       —       —       

mAP 0.184 | macro P 0.279 R 0.274 F1 0.273 @2 m | mean centre error (TP@2m) 0.648 m | 1010 frame

![alt text](docs/img/train/train0-result.png)

![alt text](docs/img/train/train0-example.png)


### 2) MATCHER = "igev"


class         AP@0.5  AP@1    AP@2    AP@4      mean   n_gt
-----------------------------------------------------------
VEHICLE       0.200   0.406   0.521   0.551   0.419  805
PERSON        0.000   0.000   0.000   0.000   0.000  18
TWO_WHEELER   —       —       —       —       —      0
TRAFFIC_SIGN  0.118   0.147   0.163   0.170   0.150  274
TRAIN         —       —       —       —       —      0

F1-optimal operating point @2 m (apply 'confidence >= score' at deployment):
class         prec    recall  F1      score   
----------------------------------------------
VEHICLE       0.592   0.506   0.545   0.200   
PERSON        0.000   0.000   0.000   nan     
TWO_WHEELER   —       —       —       —       
TRAFFIC_SIGN  0.259   0.380   0.308   0.110   
TRAIN         —       —       —       —       

mAP 0.190 | macro P 0.284 R 0.295 F1 0.284 @2 m | mean centre error (TP@2m) 0.683 m | 1010 frames

![alt text](docs/img/train/train1-loss.png)
![alt text](docs/img/train/train1-result.png)
![alt text](docs/img/train/train1-example.png)
![alt text](docs/img/train/train1-example2.png)
![alt text](docs/img/train/train1-example3.png)

Issue with encoder nms
![alt text](docs/img/train/train1-example4.png)


# 3) Efficient Network, nms fix

fix: nms decoder
new file organization
remove TRAIN class

CameraOnlyDetector: 1,066,790 trainable
Note: more trainable parameters, but no tranferlearning
TODO: test also unfreezed yolo (maybe not the whole network)


# 4) Validation sweep — dropout / depth-pre-BEV / yolo-unfreeze (2026-07-07 AM)

Quick A/B to validate the three morning choices before the full runs. **NOT
final numbers.** Fast-validation protocol: named split (train 0003+0007+0009,
val 0010) **temporally sub-sampled** to stay at LOG scale — `train_stride=8`
(2,143 frames) / `val_stride=4` (757), 5 epochs, SEED=0, MATCHER=igev, NMS on,
single seed. KITTI-360 is 10 Hz so stride-8 barely loses information but keeps a
run ~11 min (eff) / ~14 min (yolo). Single-run gaps of ±0.02–0.03 mAP are within
noise (tiny data, one seed) — read these as *direction*, not verdicts.

Two baselines + one experiment each (change one variable):

| run                     | trainable | mAP    | VEH mean / AP@2 | SIGN mean | centre-err (TP@2m) | best ep |
|-------------------------|-----------|--------|-----------------|-----------|--------------------|---------|
| base_eff (efficientnet) | 1,066,790 | 0.105  | 0.391 / 0.522   | 0.027     | 0.730 m            | 4       |
| dropout_eff (p=0.2, wd=1e-4) | 1,066,790 | 0.095 | 0.363 / 0.491 | 0.018 | 0.736 m       | 4       |
| depth_eff (refine_depth)| 1,085,158 | 0.106  | 0.391 / 0.506   | 0.033     | **0.674 m**        | 4       |
| base_yolo (frozen)      |   771,974 | **0.188** | 0.464 / 0.585 | 0.063   | 0.627 m            | 4       |
| unfreeze_yolo (last-4)  |   926,086 | 0.184  | 0.428 / 0.540   | **0.085** | 0.622 m            | 3       |

(mean = mean AP over 0.5/1/2/4 m; PERSON / TWO_WHEELER ≈ 0 GT-sparse, omitted.)

**Backbone (context):** frozen **yolo26 ≫ efficientnet** (mAP 0.188 vs 0.105) —
the COCO-pretrained stem is the clearly stronger camera branch, as expected.

**1) Dropout (test dropout).** `HEAD_DROPOUT=0.2 + WEIGHT_DECAY=1e-4` on the
efficientnet baseline: mAP 0.095 vs 0.105, VEH 0.363 vs 0.391 — a small
**regression**. 5 epochs on 2 k frames don't overfit hard enough for this
strength to pay; it just slows the head. Plumbing works (Lorenzo's Dropout2d +
AdamW wd). Verdict: **don't adopt at this strength**; re-try lighter (p≈0.1, wd
1e-3) at full scale where overfitting is real (best-val was ~epoch 2 on the big
runs).

**2) Depth pre-BEV (add depth before BEV).** New `DepthContextNet` (config
`refine_depth`, +18 k params) injects the stereo depth into the *context
features* before the splat. NB the BEV splat positions are non-differentiable
(hard `.long()` cell index) — a depth-*position* refiner gets zero gradient, so
this injects depth on the differentiable **feature** path instead. Result: mAP
≈ flat (0.106 vs 0.105), but **centre error 0.674 vs 0.730 m (−8 %)** and
SIGN 0.033 vs 0.027 — depth tightens localisation, which is exactly what
grounded stereo should buy. Cheap and directionally right → **keep for the full
run** (best judged with the yolo26 backbone).

**3) YOLO unfrozen (last layers).** `YOLO_UNFREEZE_LAST=4` unfreezes the two
C3k2 blocks that **produce P3** (the tapped feature; anchored on `head.f[0]`, not
the discarded P4/P5 tail) → +154 k trainable. mAP wash (0.184 vs 0.188), trades
**VEH down (0.428 vs 0.464) for SIGN up (0.085 vs 0.063, +35 %)**, and overfits
earlier (best ep 3 vs 4). The extra capacity helps small objects at a small cost
to cars. Marginal at 5 epochs/2 k frames → **revisit at full scale** (more data
should let the unfrozen layers help without overfitting).

Reproduce: notebook knobs `HEAD_DROPOUT` / `WEIGHT_DECAY` / `REFINE_DEPTH` /
`CAMERA_BACKBONE` + `YOLO_UNFREEZE_LAST` (training.ipynb §2). Runs auto-saved to
`runs/<tag>_<ts>/` (git-ignored).

