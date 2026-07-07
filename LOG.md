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

