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


### 3) Efficient Network, nms fix
/home/lorenzo/Desktop/repo/AIRO/stereo-lidar-perception/runs/camera_efficientnet_igev_20260707_033656

fix: nms decoder
new file organization
remove TRAIN class

CameraOnlyDetector: 1,066,790 trainable
Note: more trainable parameters, but no tranferlearning
TODO: test also unfreezed yolo (maybe not the whole network)

class         AP@0.5  AP@1    AP@2    AP@4      mean   n_gt
-----------------------------------------------------------
VEHICLE       0.101   0.275   0.420   0.467   0.316  805
PERSON        0.000   0.000   0.000   0.000   0.000  18
TWO_WHEELER   —       —       —       —       —      0
TRAFFIC_SIGN  0.067   0.105   0.130   0.140   0.110  274

F1-optimal operating point @2 m (apply 'confidence >= score' at deployment):
class         prec    recall  F1      score   
----------------------------------------------
VEHICLE       0.616   0.448   0.519   0.135   
PERSON        0.000   0.000   0.000   0.161   
TWO_WHEELER   —       —       —       —       
TRAFFIC_SIGN  0.237   0.223   0.230   0.218   

mAP 0.142 | macro P 0.284 R 0.224 F1 0.250 @2 m | mean centre error (TP@2m) 0.707 m | 1010 frames

/home/lorenzo/Desktop/repo/AIRO/stereo-lidar-perception/runs/camera_efficientnet_igev_20260707_155342/plots/evaluation.png
/home/lorenzo/Desktop/repo/AIRO/stereo-lidar-perception/runs/camera_efficientnet_igev_20260707_155342/plots/loss_curves.png

### 4) Yolo p3p4
/home/lorenzo/Desktop/repo/AIRO/stereo-lidar-perception/runs/camera_yolo26_igev_20260707_162407

CameraOnlyDetector: 804,742 trainable | 2,572,280 frozen (yolo26 backbone)

fix nms in visualization

class         AP@0.5  AP@1    AP@2    AP@4      mean   n_gt
-----------------------------------------------------------
VEHICLE       0.189   0.458   0.606   0.648   0.475  805
PERSON        0.000   0.000   0.000   0.000   0.000  18
TWO_WHEELER   —       —       —       —       —      0
TRAFFIC_SIGN  0.166   0.202   0.212   0.215   0.199  274

### 4) WEIGHT_DECAY = 1e-4 (AdamW, dropout still 0)

Phase 1 regularization A/B: only weight decay on (head_dropout=0), yolo26 backbone frozen, MATCHER=igev, YOLO_LEVELS=p3. Dataset pinned to same drives as entry 1 (kitti360_train=0007 (2,890) / kitti360_val=0003 (1,010)).

CameraOnlyDetector: 804,742 trainable | 2,572,280 frozen (yolo26 backbone)

![alt text](docs/img/train/train2-loss.png)

class         AP@0.5  AP@1    AP@2    AP@4      mean   n_gt
-----------------------------------------------------------
VEHICLE       0.205   0.431   0.573   0.626   0.459  805
PERSON        0.000   0.000   0.000   0.000   0.000  18
TWO_WHEELER   —       —       —       —       —      0
TRAFFIC_SIGN  0.123   0.153   0.166   0.179   0.155  274

F1-optimal operating point @2 m (apply 'confidence >= score' at deployment):
class         prec    recall  F1      score   
----------------------------------------------
VEHICLE       0.730   0.548   0.626   0.199   
PERSON        0.000   0.000   0.000   0.153   
TWO_WHEELER   —       —       —       —       
TRAFFIC_SIGN  0.355   0.354   0.355   0.214   

mAP 0.225 | macro P 0.362 R 0.301 F1 0.327 @2 m | mean centre error (TP@2m) 0.617 m | 1010 frames



### 4) FIX NMS

bigger nsm for vehicles
class         AP@0.5  AP@1    AP@2    AP@4      mean   n_gt
-----------------------------------------------------------
VEHICLE       0.187   0.451   0.603   0.650   0.473  805
PERSON        0.000   0.000   0.000   0.000   0.000  18
TWO_WHEELER   —       —       —       —       —      0
TRAFFIC_SIGN  0.166   0.202   0.212   0.215   0.199  274

F1-optimal operating point @2 m (apply 'confidence >= score' at deployment):
class         prec    recall  F1      score   
----------------------------------------------
VEHICLE       0.747   0.544   0.630   0.199   
PERSON        0.000   0.000   0.000   0.153   
TWO_WHEELER   —       —       —       —       
TRAFFIC_SIGN  0.355   0.354   0.355   0.214   

mAP 0.224 | macro P 0.368 R 0.299 F1 0.328 @2 m | mean centre error (TP@2m) 0.614 m | 1010 frames

![alt text](docs/img/traintest-example.png)


### 5) yolo p3p4p5

CameraOnlyDetector: 870,278 trainable | 2,572,280 frozen (yolo26 backbone)

class         AP@0.5  AP@1    AP@2    AP@4      mean   n_gt
-----------------------------------------------------------
VEHICLE       0.179   0.461   0.656   0.680   0.494  805
PERSON        0.000   0.000   0.000   0.000   0.000  18
TWO_WHEELER   —       —       —       —       —      0
TRAFFIC_SIGN  0.110   0.148   0.158   0.181   0.149  274

F1-optimal operating point @2 m (apply 'confidence >= score' at deployment):
class         prec    recall  F1      score   
----------------------------------------------
VEHICLE       0.713   0.643   0.677   0.180   
PERSON        0.000   0.000   0.000   0.112   
TWO_WHEELER   —       —       —       —       
TRAFFIC_SIGN  0.242   0.369   0.292   0.185   

mAP 0.215 | macro P 0.318 R 0.337 F1 0.323 @2 m | mean centre error (TP@2m) 0.648 m | 1010 frames

Better for vehicles, worst for small objects
Maybe keep only p3p4


### 5) MONO BEV

onoOnlyDetector: 1,372,847 trainable | 2,572,280 frozen (yolo26 backbone)

class         AP@0.5  AP@1    AP@2    AP@4      mean   n_gt
-----------------------------------------------------------
VEHICLE       0.016   0.086   0.252   0.435   0.197  805
PERSON        0.000   0.000   0.000   0.000   0.000  18
TWO_WHEELER   —       —       —       —       —      0
TRAFFIC_SIGN  0.000   0.000   0.000   0.001   0.000  274

F1-optimal operating point @2 m (apply 'confidence >= score' at deployment):
class         prec    recall  F1      score   
----------------------------------------------
VEHICLE       0.437   0.348   0.387   0.145   
PERSON        0.000   0.000   0.000   nan     
TWO_WHEELER   —       —       —       —       
TRAFFIC_SIGN  0.000   0.000   0.000   0.103   

mAP 0.066 | macro P 0.146 R 0.116 F1 0.129 @2 m | mean centre error (TP@2m) 1.000 m | 1010 frames


#### 6) DEPTH + P3P4

CameraOnlyDetector: 809,350 trainable | 2,572,280 frozen (yolo26 backbone)

class         AP@0.5  AP@1    AP@2    AP@4      mean   n_gt
-----------------------------------------------------------
VEHICLE       0.123   0.424   0.644   0.684   0.469  805
PERSON        0.002   0.002   0.002   0.002   0.002  18
TWO_WHEELER   #### 6) DEPTH + P3P4—       —       —       —       —      0
TRAFFIC_SIGN  0.124   0.174   0.185   0.200   0.171  274

F1-optimal operating point @2 m (apply 'confidence >= score' at deployment):
class         prec    recall  F1      score   
----------------------------------------------
VEHICLE       0.742   0.583   0.653   0.184   
PERSON        0.043   0.056   0.049   0.102   
TWO_WHEELER   —       —       —       —       
TRAFFIC_SIGN  0.316   0.310   0.313   0.282   

mAP 0.214 | macro P 0.367 R 0.316 F1 0.338 @2 m | mean centre error (TP@2m) 0.673 m | 1010 frames
VEHICLE       0.619   0.645   0.632   0.104   
PERSON        0.000   0.000   0.000   0.117   
TWO_WHEELER   —       —       —       —       
TRAFFIC_SIGN  0.318   0.376   0.344   0.129   

mAP 0.205 | macro P 0.312 R 0.340 F1 0.325 @2 m | mean centre error (TP@2m) 0.593 m | 1010 frames

![alt text](docs/img/train/train2-result.png)

![alt text](docs/img/train/train2-example.png)

#### 6) DEPTH + P3P4 + AdamW + Dropout

WEIGHT_DECAY = 1e-4
HEAD_DROPOUT = 0.1

CameraOnlyDetector: 809,350 trainable | 2,572,280 frozen (yolo26 backbone)

class         AP@0.5  AP@1    AP@2    AP@4      mean   n_gt
-----------------------------------------------------------
VEHICLE       0.184   0.448   0.631   0.688   0.488  805
PERSON        0.007   0.008   0.008   0.008   0.007  18
TWO_WHEELER   —       —       —       —       —      0
TRAFFIC_SIGN  0.153   0.184   0.196   0.211   0.186  274

F1-optimal operating point @2 m (apply 'confidence >= score' at deployment):
class         prec    recall  F1      score   
----------------------------------------------
VEHICLE       0.761   0.573   0.653   0.190   
PERSON        0.029   0.222   0.052   0.109   
TWO_WHEELER   —       —       —       —       
TRAFFIC_SIGN  0.323   0.376   0.347   0.200   

mAP 0.227 | macro P 0.371 R 0.390 F1 0.351 @2 m | mean centre error (TP@2m) 0.613 m | 1010 frames

#### 7) DEPTH + P3 (ignore it!)

class         AP@0.5  AP@1    AP@2    AP@4      mean   n_gt
-----------------------------------------------------------
VEHICLE       0.172   0.454   0.633   0.677   0.484  805
PERSON        0.000   0.000   0.000   0.000   0.000  18
TWO_WHEELER   —       —       —       —       —      0
TRAFFIC_SIGN  0.092   0.132   0.134   0.147   0.126  274

F1-optimal operating point @2 m (apply 'confidence >= score' at deployment):
class         prec    recall  F1      score   
----------------------------------------------
VEHICLE       0.703   0.625   0.661   0.206   
PERSON        0.000   0.000   0.000   0.132   
TWO_WHEELER   —       —       —       —       
TRAFFIC_SIGN  0.241   0.318   0.274   0.205   

mAP 0.204 | macro P 0.315 R 0.314 F1 0.312 @2 m | mean centre error (TP@2m) 0.651 m | 1010 frames

![alt text](docs/img/train/train3-result.png)
![alt text](docs/img/train/train3-example.png)



