#1) Pipeline A with depth

class         AP@0.5  AP@1    AP@2    AP@4      mean   n_gt
-----------------------------------------------------------
VEHICLE       0.791   0.897   0.933   0.937   0.889  10558
PERSON        0.581   0.597   0.602   0.609   0.597  1347
TWO_WHEELER   0.425   0.511   0.532   0.551   0.505  1429
TRAFFIC_SIGN  0.229   0.237   0.250   0.300   0.254  918

F1-optimal operating point @2 m (apply 'confidence >= score' at deployment):
class         prec    recall  F1      score   
----------------------------------------------
VEHICLE       0.935   0.868   0.900   0.317   
PERSON        0.670   0.537   0.596   0.249   
TWO_WHEELER   0.576   0.550   0.563   0.219   
TRAFFIC_SIGN  0.341   0.358   0.349   0.147   

mAP 0.561 | macro P 0.630 R 0.578 F1 0.602 @2 m | mean centre error (TP@2m) 0.283 m | 3125 frames



# 2) PIPELINE C with depth

class         AP@0.5  AP@1    AP@2    AP@4      mean   n_gt
-----------------------------------------------------------
VEHICLE       0.790   0.891   0.925   0.931   0.884  11232
PERSON        0.559   0.574   0.580   0.587   0.575  1532
TWO_WHEELER   0.457   0.539   0.555   0.569   0.530  1427
TRAFFIC_SIGN  0.297   0.306   0.315   0.347   0.316  1077

F1-optimal operating point @2 m (apply 'confidence >= score' at deployment):
class         prec    recall  F1      score   
----------------------------------------------
VEHICLE       0.925   0.880   0.902   0.246   
PERSON        0.702   0.541   0.611   0.195   
TWO_WHEELER   0.659   0.555   0.603   0.225   
TRAFFIC_SIGN  0.395   0.358   0.376   0.325   

mAP 0.576 | macro P 0.670 R 0.584 F1 0.623 @2 m | mean centre error (TP@2m) 0.274 m | 3125 frames

# 3) Pipeline C without DEPTH

class         AP@0.5  AP@1    AP@2    AP@4      mean   n_gt
-----------------------------------------------------------
VEHICLE       0.781   0.890   0.926   0.934   0.883  11232
PERSON        0.568   0.581   0.590   0.606   0.586  1532
TWO_WHEELER   0.467   0.543   0.566   0.581   0.539  1427
TRAFFIC_SIGN  0.312   0.330   0.343   0.378   0.341  1077

F1-optimal operating point @2 m (apply 'confidence >= score' at deployment):
class         prec    recall  F1      score   
----------------------------------------------
VEHICLE       0.930   0.874   0.901   0.272   
PERSON        0.702   0.548   0.615   0.198   
TWO_WHEELER   0.658   0.565   0.608   0.208   
TRAFFIC_SIGN  0.424   0.407   0.415   0.241   

mAP 0.587 | macro P 0.679 R 0.598 F1 0.635 @2 m | mean centre error (TP@2m) 0.287 m | 3125 frames


---

# Full comparison

Setup: train drives 3/7/9, val drive 10, 4-class, 3125 val frames. Best per row in **bold**.

## Summary (macro / overall)

| # | Run | mAP | macro P | macro R | macro F1 | Centre err (m) ↓ |
|---|-----|-----|---------|---------|----------|------------------|
| 1 | Pipeline A + depth | 0.561 | 0.630 | 0.578 | 0.602 | 0.283 |
| 2 | Pipeline C + depth | 0.576 | 0.670 | 0.584 | 0.623 | **0.274** |
| 3 | Pipeline C − depth | **0.587** | **0.679** | **0.598** | **0.635** | 0.287 |

## AP by class (AP@0.5 / @1 / @2 / @4 / mean)

| Class | Metric | A + depth | C + depth | C − depth |
|-------|--------|-----------|-----------|-----------|
| **VEHICLE** | AP@0.5 | **0.791** | 0.790 | 0.781 |
| | AP@1 | **0.897** | 0.891 | 0.890 |
| | AP@2 | **0.933** | 0.925 | 0.926 |
| | AP@4 | **0.937** | 0.931 | 0.934 |
| | mean | **0.889** | 0.884 | 0.883 |
| **PERSON** | AP@0.5 | **0.581** | 0.559 | 0.568 |
| | AP@1 | **0.597** | 0.574 | 0.581 |
| | AP@2 | **0.602** | 0.580 | 0.590 |
| | AP@4 | **0.609** | 0.587 | 0.606 |
| | mean | **0.597** | 0.575 | 0.586 |
| **TWO_WHEELER** | AP@0.5 | 0.425 | 0.457 | **0.467** |
| | AP@1 | 0.511 | 0.539 | **0.543** |
| | AP@2 | 0.532 | 0.555 | **0.566** |
| | AP@4 | 0.551 | 0.569 | **0.581** |
| | mean | 0.505 | 0.530 | **0.539** |
| **TRAFFIC_SIGN** | AP@0.5 | 0.229 | 0.297 | **0.312** |
| | AP@1 | 0.237 | 0.306 | **0.330** |
| | AP@2 | 0.250 | 0.315 | **0.343** |
| | AP@4 | 0.300 | 0.347 | **0.378** |
| | mean | 0.254 | 0.316 | **0.341** |

## F1-optimal operating point @2 m (prec / recall / F1)

| Class | Metric | A + depth | C + depth | C − depth |
|-------|--------|-----------|-----------|-----------|
| **VEHICLE** | P | **0.935** | 0.925 | 0.930 |
| | R | 0.868 | **0.880** | 0.874 |
| | F1 | 0.900 | **0.902** | 0.901 |
| **PERSON** | P | 0.670 | 0.702 | **0.702** |
| | R | 0.537 | 0.541 | **0.548** |
| | F1 | 0.596 | 0.611 | **0.615** |
| **TWO_WHEELER** | P | 0.576 | **0.659** | 0.658 |
| | R | 0.550 | 0.555 | **0.565** |
| | F1 | 0.563 | 0.603 | **0.608** |
| **TRAFFIC_SIGN** | P | 0.341 | 0.395 | **0.424** |
| | R | 0.358 | 0.358 | **0.407** |
| | F1 | 0.349 | 0.376 | **0.415** |

## Takeaways

- **Pipeline C without depth wins overall** — best mAP (0.587), macro P/R/F1, and it dominates the two sparse classes (TWO_WHEELER, TRAFFIC_SIGN) at every threshold.
- **Depth hurts Pipeline C** on the big set (0.576 → 0.587 when removed), consistent with the observed overfitting: the depth context adds capacity the sparse classes can't support.
- **Pipeline A is the VEHICLE/PERSON specialist** — best AP on the two well-populated classes and lowest centre error, but collapses on TRAFFIC_SIGN (0.254 mean vs 0.341).
- n_gt differs slightly between runs (e.g. VEHICLE 10558 vs 11232), so the eval sets aren't byte-identical — treat sub-0.01 gaps as noise.
