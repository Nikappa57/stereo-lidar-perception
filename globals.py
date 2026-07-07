"""Project-wide configuration — the single source of truth.

Everything that has to agree across the two branches and the fusion lives here:
the shared **BEV grid** (extent + resolution), the **channel contract** between
Stage A and the fusion, and the **class set**. Every per-module config dataclass
(`PillarConfig`, `MonoBEVConfig`, `StereoBEVConfig`, `BEVFusionConfig`, …)
defaults to these constants instead of re-declaring the numbers, so the grid and
the 64/128 channel contract can never drift between modules (design doc §02; was
previously duplicated across four configs — see TODO P0).

The shared grid::

    x ∈ [0, 50] m (forward), y ∈ [-20, 20] m (lateral), z ∈ [-3, 1] m
    0.25 m / cell  ->  nx = 200, ny = 160      (indexing: flat = ix * ny + iy)

Note: this module is named ``globals.py`` to match the project layout. It does
**not** shadow the builtin ``globals()`` function — that is only reachable by
name, and ``import globals`` binds the module, not the builtin.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# Shared BEV grid (must be identical across every branch and the fusion)
# --------------------------------------------------------------------------- #
X_RANGE: tuple[float, float] = (0.0, 50.0)  # forward (ego +x)
Y_RANGE: tuple[float, float] = (-20.0, 20.0)  # lateral (ego +y)
Z_RANGE: tuple[float, float] = (-3.0, 1.0)  # height pre-filter
BEV_RES_M: float = 0.25  # metres per BEV cell


def grid_size(
    x_range: tuple[float, float] = X_RANGE,
    y_range: tuple[float, float] = Y_RANGE,
    res_m: float = BEV_RES_M,
) -> tuple[int, int]:
    """`(nx, ny)` cell counts for a BEV extent + resolution. Defaults = shared grid."""
    nx = int(round((x_range[1] - x_range[0]) / res_m))
    ny = int(round((y_range[1] - y_range[0]) / res_m))
    return nx, ny


GRID_SIZE: tuple[int, int] = grid_size()  # (200, 160) with defaults

# --------------------------------------------------------------------------- #
# Channel contract between Stage A (the two branches) and the fusion + head
# --------------------------------------------------------------------------- #
CAMERA_BEV_CHANNELS: int = 64  # C_cam — StereoBEV/MonoBEV context width
LIDAR_BEV_CHANNELS: int = 128  # C_lidar — PointPillars BEVBackbone2D output
FUSED_CHANNELS: int = 128  # fused feature depth fed to the head

# --------------------------------------------------------------------------- #
# Dataset selection — single source of truth (flip these to swap datasets)
# --------------------------------------------------------------------------- #
# The pipeline is dataset-agnostic (everything goes through py123d's SceneAPI);
# only the split names + stereo camera ids change between datasets. KITTI-360 is
# the primary dataset because it is the only py123d source with a real *colour*
# stereo pair (AV2's stereo pair is grayscale). See docs/dataset.md.
DATASET: str = "kitti360"
TRAIN_SPLIT: str = "kitti360_train"
VAL_SPLIT: str = "kitti360_val"
# Stereo pair camera ids. KITTI-360 maps image_00/image_01 → these same enums,
# so the id names are shared with AV2; only the split names above differ.
LEFT_CAMERA: str = "pcam_stereo_l"
RIGHT_CAMERA: str = "pcam_stereo_r"

# --------------------------------------------------------------------------- #
# Classes (design doc §05)
# --------------------------------------------------------------------------- #
# The py123d loader emits a *unified* taxonomy (via ``default_label``), coarser
# than each dataset's raw names. We train on **every unified class KITTI-360
# actually produces**. The raw KITTI-360 → unified folding is (parser registry):
#   VEHICLE        ← car, truck, bus, caravan, trailer
#   PERSON         ← pedestrian, rider
#   TWO_WHEELER    ← bicycle, motorcycle
#   TRAFFIC_SIGN   ← trafficSign, stop
#   TRAIN          ← train
#   GENERIC_OBJECT ← bigPole, smallPole, lamp, box, trashbin, vendingmachine
# NOTE: KITTI-360's ``trafficLight`` exists in the XML (~147 boxes) but py123d's
# parser does **not** emit it — verified 0 TRAFFIC_LIGHT detections on the val
# drive (which has 74 in the raw XML). So TRAFFIC_LIGHT is intentionally omitted:
# a class with zero GT only wastes a head channel and a permanently-0 AP row.
# Class-frequency notes (measured on drives 0003/0007/0009/0010, see docs):
#   * GENERIC_OBJECT (bigPole/smallPole/lamp/box/trashbin/vendingmachine) is a
#     heterogeneous, noisy bucket (mostly thin poles) — **excluded**: those raw
#     labels now fall into the ignore bucket (class_index -> None).
#   * TRAIN is emitted but rare and only on drive 0010 → **excluded**: with 0010
#     as the val split it never appears in training (0 GT on drives 0003/0007/
#     0009), so a TRAIN channel only wastes head capacity on a permanently-0 AP.
# TRAFFIC_CONE (the Formula-Student deployment target) does not exist in
# KITTI-360; it must come from AV2 (CONSTRUCTION_CONE) or CARLA later.
CLASSES: tuple[str, ...] = (
    "VEHICLE",
    "PERSON",
    "TWO_WHEELER",
    "TRAFFIC_SIGN",
)
NUM_CLASSES: int = len(CLASSES)

# Loader label -> training class index. Labels absent here are ignored (not a
# negative, not a positive — dropped before encoding/loss).
CLASS_REMAP: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}


def class_index(label: str) -> int | None:
    """Map a loader label to its training class index, or ``None`` to ignore it."""
    return CLASS_REMAP.get(label)
